#!/usr/bin/env python3
"""
NIM-OV-Bridge v3: Automated NIM Container -> Intel iGPU Bridge
Extracts model weights FROM the NIM container itself, no HF re-download needed.

Usage:
    # With existing local NIM image (extracts weights from container):
    python3 nim_ov_bridge.py \
        --container-url nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
        --ngc-api-key $NGC_API_KEY \
        --model-name llama-3.1-8b \
        --port 8000 \
        --quantize int8 \
        --use-existing-image

    # With HF fallback (if extraction fails or you want a different model):
    python3 nim_ov_bridge.py \
        --container-url nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
        --ngc-api-key $NGC_API_KEY \
        --model-name llama-3.1-8b \
        --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --port 8000 \
        --quantize int8
"""

import argparse
import os
import sys
import subprocess
import json
import shutil
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = Path.home() / ".nim_ov_bridge" / "models"
BUILD_DIR = Path.home() / ".nim_ov_bridge" / "build"


def run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None, check: bool = True):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=check, capture_output=False, text=True)
    return result


def check_prerequisites():
    print("[CHECK] Verifying host prerequisites...")
    container_cmd = None
    for cmd in ["docker", "podman"]:
        if shutil.which(cmd):
            container_cmd = cmd
            break
    if not container_cmd:
        print("[ERROR] Docker or Podman not found.")
        sys.exit(1)
    print(f"[OK] Container runtime: {container_cmd}")

    dri_path = Path("/dev/dri")
    if dri_path.exists():
        render_nodes = list(dri_path.glob("renderD*"))
        if render_nodes:
            print(f"[OK] Intel GPU render nodes found: {render_nodes}")
    try:
        import optimum
        print("[OK] optimum installed")
    except ImportError:
        print("[WARN] optimum not installed. Run: pip install optimum[openvino]")
    return container_cmd


def ngc_login(container_url: str, api_key: str):
    print("[NGC] Logging into NVIDIA Container Registry...")
    registry = container_url.split("/")[0]
    cmd = ["docker", "login", registry, "-u", "$oauthtoken", "--password-stdin"]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, input=api_key, text=True, capture_output=False, check=False)
    if result.returncode != 0:
        print(f"[WARN] Docker login to {registry} failed.")


def pull_or_inspect_image(container_url: str, existing: bool = False) -> str:
    if existing:
        print(f"[IMAGE] Checking for existing image or container: {container_url}")
        
        # 1. Check if it's an existing container
        result = subprocess.run(["docker", "inspect", "--type=container", "-f", "{{.Id}}", container_url],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            container_id = result.stdout.strip()
            print(f"[OK] Found existing container: {container_id[:12]}")
            return container_url

        # 2. Check if it's an image
        result = subprocess.run(["docker", "inspect", "--type=image", "-f", "{{.Id}}", container_url],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            image_id = result.stdout.strip()
            print(f"[OK] Found existing image: {image_id[:19]}")
            return container_url

        # 3. Fallback: Check using docker images (handles some tagged registries better)
        result = subprocess.run(["docker", "images", "-q", container_url],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            image_id = result.stdout.strip().split('\n')[0]
            print(f"[OK] Found existing image via docker images: {image_id}")
            return container_url

        print(f"[WARN] Image or container {container_url} not found locally. Proceeding to pull...")

    print(f"[IMAGE] Pulling NIM container: {container_url}")
    run(["docker", "pull", container_url])
    return container_url


def inspect_nim_structure(image_id: str) -> dict:
    print("[INSPECT] Analyzing NIM container structure...")
    result = subprocess.run(["docker", "inspect", image_id],
                            capture_output=True, text=True, check=True)
    inspect_data = json.loads(result.stdout)[0]
    config = inspect_data.get("Config", {})
    info = {
        "entrypoint": config.get("Entrypoint", []),
        "cmd": config.get("Cmd", []),
        "working_dir": config.get("WorkingDir", "/opt/nim"),
        "env": config.get("Env", []),
        "image_id": image_id,
    }
    print(f"[INFO] Original entrypoint: {info['entrypoint']}")
    print(f"[INFO] Working dir: {info['working_dir']}")
    return info


def extract_model_from_nim(image_id: str, model_name: str) -> Optional[Path]:
    """
    Extract model weights FROM the NIM container itself.
    NIM containers typically store weights in /model-repo, /models, /opt/nim/model-store,
    or as HuggingFace format in /model-store/hf_models/.
    """
    print(f"[EXTRACT] Attempting to extract model weights from: {image_id}")
    extract_dir = MODELS_DIR / model_name / "nim_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Common NIM model paths inside containers
    possible_paths = [
        "/model-store",           # NIM standard model store
        "/models",                # Generic models dir
        "/opt/nim/model-store",   # NIM-specific path
        "/opt/nim/models",        # Alternative NIM path
        "/model-repo",            # Triton model repository
        "/workspace/models",      # Some NIM variants
    ]

    # Check if image_id is actually a container
    result = subprocess.run(["docker", "inspect", "--type=container", "-f", "{{.Id}}", image_id],
                            capture_output=True, text=True, check=False)
    is_container = (result.returncode == 0 and result.stdout.strip() != "")

    if is_container:
        print(f"[EXTRACT] Source is a container. Extracting directly from container '{image_id}'...")
        temp_container = image_id
        created_temp = False
    else:
        # Create a temporary container to extract files (don't run it)
        temp_container = f"nim-extract-temp-{model_name.replace('/', '-')}"
        run(["docker", "create", "--name", temp_container, image_id, "true"], check=False)
        created_temp = True

    extracted_path = None

    try:
        for path in possible_paths:
            print(f"[EXTRACT] Checking {path} in container...")
            # List contents to see if path exists
            ls_result = subprocess.run(
                ["docker", "cp", f"{temp_container}:{path}", "-"],
                capture_output=True, text=True, check=False
            )
            if ls_result.returncode == 0 or "tar" in str(ls_result.stderr):
                # Path exists, extract it
                dest = extract_dir / Path(path).name
                print(f"[EXTRACT] Found model data at {path}, extracting to {dest}...")
                try:
                    run(["docker", "cp", f"{temp_container}:{path}", str(dest)], check=False)
                    # For NIM containers, paths might exist but be empty if weights weren't downloaded
                    if dest.exists() and any(dest.iterdir()):
                        extracted_path = dest
                        print(f"[OK] Extracted model data to {extracted_path}")
                        break
                    else:
                        print(f"[WARN] Extracted path {dest} is empty. Skipping...")
                except Exception as e:
                    print(f"[WARN] Failed to extract {path}: {e}")
                    continue

        if not extracted_path:
            print("[WARN] Could not find standard model paths. Scanning entire container...")
            if not is_container:
                # Try to find any .safetensors, .bin, or config.json files excluding examples/tests
                find_result = subprocess.run(
                    ["docker", "run", "--rm", "--entrypoint", "sh", image_id,
                     "-c", "find / -type f \\( -name '*.safetensors' -o -name 'pytorch_model.bin' -o -name 'config.json' \\) -not -path '*/examples/*' -not -path '*/tests/*' 2>/dev/null | head -5"],
                    capture_output=True, text=True, check=False
                )
                if find_result.stdout:
                    print(f"[INFO] Found model files:\n{find_result.stdout}")
                    # Try extracting the parent directory of first found file
                    first_file = find_result.stdout.strip().split('\n')[0]
                    parent_dir = str(Path(first_file).parent)
                    dest = extract_dir / "discovered"
                    print(f"[EXTRACT] Extracting {parent_dir}...")
                    run(["docker", "cp", f"{temp_container}:{parent_dir}", str(dest)], check=False)
                    if dest.exists():
                        extracted_path = dest
            else:
                print("[WARN] Advanced scanning is not supported for running containers.")
    finally:
        # Clean up temp container
        if created_temp:
            run(["docker", "rm", "-f", temp_container], check=False)

    if extracted_path and extracted_path.exists():
        # Look for actual HF-format model inside extracted data
        # NIM often nests HF models inside subdirectories
        hf_candidates = list(extracted_path.rglob("config.json"))
        if hf_candidates:
            # The model weights are in the same dir as config.json
            model_root = hf_candidates[0].parent
            print(f"[OK] Found HuggingFace-format model at: {model_root}")
            return model_root
        else:
            # Return the extracted path as-is; conversion will handle it
            return extracted_path

    print("[ERROR] Could not extract model from NIM container.")
    return None


def convert_to_openvino(source_path: Path, model_name: str, quantize: str = "int8",
                        hf_model_id: Optional[str] = None) -> Path:
    """
    Convert model to OpenVINO IR. Uses local source_path first, falls back to HF if provided.
    """
    model_out = MODELS_DIR / model_name / "openvino"
    if (model_out / "openvino_model.xml").exists():
        print(f"[MODEL] OpenVINO IR already exists at {model_out}")
        return model_out

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Try local conversion first
    if source_path and source_path.exists():
        print(f"[MODEL] Converting local model from {source_path} to OpenVINO IR ({quantize})...")
        return _convert_local(source_path, model_out, quantize)

    # Fallback to HF if provided
    if hf_model_id:
        print(f"[MODEL] Falling back to HuggingFace: {hf_model_id}")
        return _convert_from_hf(hf_model_id, model_out, quantize)

    print("[ERROR] No local model found and no --hf-model-id provided.")
    sys.exit(1)


def _convert_local(source_path: Path, output_dir: Path, quantize: str) -> Path:
    """Convert local HF-format model directory to OpenVINO IR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Local conversion: {source_path} -> {output_dir}")

    # Check if source is already OpenVINO IR
    if (source_path / "openvino_model.xml").exists():
        print("[MODEL] Source is already OpenVINO IR, copying...")
        for f in source_path.glob("openvino_model.*"):
            shutil.copy(f, output_dir)
        for f in source_path.glob("*.json"):
            shutil.copy(f, output_dir)
        return output_dir

    # Use optimum-cli for local export
    cmd = [
        "python3", "-m", "optimum_cli", "export", "openvino",
        "--model", str(source_path),
        "--task", "text-generation-with-past",
        "--weight-format", quantize,
        str(output_dir)
    ]
    try:
        run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] optimum-cli failed ({e}), falling back to Python export...")
        _fallback_local_export(source_path, output_dir, quantize)

    print(f"[OK] Model exported to: {output_dir}")
    return output_dir


def _convert_from_hf(hf_model_id: str, output_dir: Path, quantize: str) -> Path:
    """Download from HuggingFace and convert to OpenVINO IR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Downloading {hf_model_id} from HuggingFace and converting to OpenVINO IR ({quantize})...")
    print("[MODEL] This may take 10-30 minutes and requires ~20GB RAM...")

    cmd = [
        "python3", "-m", "optimum_cli", "export", "openvino",
        "--model", hf_model_id,
        "--task", "text-generation-with-past",
        "--weight-format", quantize,
        str(output_dir)
    ]
    try:
        run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] optimum-cli failed ({e}), falling back to Python export...")
        _fallback_hf_export(hf_model_id, output_dir, quantize)

    print(f"[OK] Model exported to: {output_dir}")
    return output_dir


def _fallback_local_export(source_path: Path, output_dir: Path, quantize: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(
    "{source_path}",
    export=True,
    load_in_8bit={"True" if quantize == "int8" else "False"},
    device="cpu",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained("{source_path}", local_files_only=True)
model.save_pretrained("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print("Local export complete.")
"""
    script_path = output_dir / "_export_local.py"
    script_path.write_text(script)
    run(["python3", str(script_path)])


def _fallback_hf_export(hf_model_id: str, output_dir: Path, quantize: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(
    "{hf_model_id}",
    export=True,
    load_in_8bit={"True" if quantize == "int8" else "False"},
    device="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("{hf_model_id}")
model.save_pretrained("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print("HF export complete.")
"""
    script_path = output_dir / "_export_hf.py"
    script_path.write_text(script)
    run(["python3", str(script_path)])


def build_cuda_stubs():
    stub_dir = SCRIPT_DIR / "cuda_stub"
    build_path = BUILD_DIR / "cuda_stubs"
    build_path.mkdir(parents=True, exist_ok=True)
    print("[BUILD] Compiling CUDA stub libraries...")
    shutil.copy(stub_dir / "libcuda_stub.c", build_path / "libcuda_stub.c")
    shutil.copy(stub_dir / "libcudart_stub.c", build_path / "libcudart_stub.c")
    shutil.copy(stub_dir / "Makefile", build_path / "Makefile")
    run(["make", "-j4"], cwd=build_path)
    return build_path


def generate_dockerfile(nim_image: str, model_name: str, port: int) -> Path:
    dockerfile_path = BUILD_DIR / "Dockerfile"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    template = f"""# NIM-OV-Bridge v3: Intel iGPU Patched NIM Container
FROM {nim_image}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# 1. Intel GPU Compute Runtime for Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ocl-icd-libopencl1 \\
    intel-opencl-icd \\
    libze-intel-gpu1 \\
    libze-loader \\
    libigdgmm12 \\
    clinfo \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# 2. OpenVINO Runtime + GenAI
RUN pip3 install --no-cache-dir --upgrade pip && \\
    pip3 install --no-cache-dir \\
    openvino==2025.1.0 \\
    openvino-genai==2025.1.0 \\
    openvino-tokenizers==2025.1.0 \\
    optimum-intel[openvino] \\
    optimum \\
    nncf \\
    fastapi \\
    uvicorn[standard] \\
    pydantic \\
    transformers \\
    accelerate \\
    huggingface-hub \\
    numpy \\
    tiktoken

# 3. CUDA Stub Libraries
COPY cuda_stubs/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
COPY cuda_stubs/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
COPY cuda_stubs/libcudart.so.11.0 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0
COPY cuda_stubs/libcudart.so.11.0 /usr/lib/x86_64-linux-gnu/libcudart.so
RUN ldconfig

# 4. Patched API Server
COPY patches/openvino_api_server.py /opt/nim-ov-bridge/openvino_api_server.py
COPY patches/entrypoint.sh /opt/nim-ov-bridge/entrypoint.sh
RUN chmod +x /opt/nim-ov-bridge/entrypoint.sh

# 5. Environment
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/intel/openvino/runtime/lib:$LD_LIBRARY_PATH"
ENV OPENVINO_DEVICE=GPU
ENV OV_GPU_PLUGIN=1
ENV NIM_MODEL_NAME={model_name}
ENV API_PORT={port}

# 6. Override Entrypoint
WORKDIR /opt/nim-ov-bridge
EXPOSE {port}
ENTRYPOINT ["/opt/nim-ov-bridge/entrypoint.sh"]
"""
    dockerfile_path.write_text(template)
    return dockerfile_path


def build_patched_image(dockerfile: Path, tag: str, model_path: Path):
    print(f"[BUILD] Building patched image: {tag}")
    ctx_dir = BUILD_DIR / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    stubs = BUILD_DIR / "cuda_stubs"
    shutil.copytree(stubs, ctx_dir / "cuda_stubs", dirs_exist_ok=True)

    patches = SCRIPT_DIR / "patches"
    shutil.copytree(patches, ctx_dir / "patches", dirs_exist_ok=True)

    shutil.copy(dockerfile, ctx_dir / "Dockerfile")

    run([
        "docker", "build",
        "-t", tag,
        "-f", str(ctx_dir / "Dockerfile"),
        str(ctx_dir)
    ])
    print(f"[OK] Patched image built: {tag}")


def deploy_container(image_tag: str, port: int, model_path: Path):
    print(f"[DEPLOY] Starting container on port {port}...")
    container_name = f"nim-ov-bridge-{port}"
    subprocess.run(["docker", "rm", "-f", container_name],
                   capture_output=True, check=False)

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--device", "/dev/dri:/dev/dri",
        "-p", f"{port}:{port}",
        "-v", f"{model_path}:/models:ro",
        "-e", "OPENVINO_DEVICE=GPU",
        "-e", f"MODEL_PATH=/models",
        "-e", f"API_PORT={port}",
        "--memory", "24g",
        "--shm-size", "4g",
        image_tag
    ]
    run(cmd)

    print("[DEPLOY] Waiting for API server to start (15s)...")
    time.sleep(15)

    try:
        import urllib.request
        req = urllib.request.Request(f"http://localhost:{port}/health")
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            print(f"[OK] API healthy at http://localhost:{port}/health")
    except Exception as e:
        print(f"[WARN] Health check failed: {e}")
        print(f"[INFO] Check logs: docker logs {container_name}")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                NIM-OV-Bridge v3 Deployed                     ║
╠══════════════════════════════════════════════════════════════╣
║  API Endpoint:    http://localhost:{port}/v1/chat/completions ║
║  Health Check:    http://localhost:{port}/health             ║
║  Model Path:      {model_path}          ║
║  Container:       {container_name}                    ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="NIM-OV-Bridge v3: Deploy NIM containers on Intel iGPU")
    parser.add_argument("--container-url", required=True, help="NIM container URL")
    parser.add_argument("--ngc-api-key", required=True, help="NVIDIA NGC API Key")
    parser.add_argument("--model-name", required=True, help="Model name for OpenVINO IR directory")
    parser.add_argument("--hf-model-id", default=None, help="HuggingFace model ID (fallback if NIM extraction fails)")
    parser.add_argument("--port", type=int, default=8000, help="Host port to expose API")
    parser.add_argument("--quantize", default="int8", choices=["fp16", "int8", "int4"])
    parser.add_argument("--use-existing-image", action="store_true", help="Use locally existing NIM image")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip model conversion if OpenVINO IR exists")
    parser.add_argument("--force-hf", action="store_true", help="Force HuggingFace download instead of NIM extraction")

    args = parser.parse_args()

    print("=" * 60)
    print("  NIM-OV-Bridge v3: Intel iGPU Deployment Tool")
    print("  Target: Ubuntu 22.04 LTS + i9-14900HX UHD Graphics")
    print("=" * 60)

    container_cmd = check_prerequisites()
    ngc_login(args.container_url, args.ngc_api_key)
    image_id = pull_or_inspect_image(args.container_url, existing=args.use_existing_image)
    nim_info = inspect_nim_structure(image_id)

    # Determine model source
    extracted_path = None
    if not args.force_hf and not args.skip_conversion:
        # Try extracting from NIM container first
        extracted_path = extract_model_from_nim(image_id, args.model_name)

    if not args.skip_conversion:
        ov_model_path = convert_to_openvino(
            source_path=extracted_path,
            model_name=args.model_name,
            quantize=args.quantize,
            hf_model_id=args.hf_model_id
        )
    else:
        ov_model_path = MODELS_DIR / args.model_name / "openvino"
        if not ov_model_path.exists():
            print(f"[ERROR] --skip-conversion set but {ov_model_path} not found.")
            sys.exit(1)

    stubs_path = build_cuda_stubs()
    dockerfile = generate_dockerfile(args.container_url, args.model_name, args.port)
    patched_tag = f"nim-ov-bridge/{args.model_name}:latest"
    build_patched_image(dockerfile, patched_tag, ov_model_path)
    deploy_container(patched_tag, args.port, ov_model_path)

    print("[DONE] Deployment complete.")


if __name__ == "__main__":
    main()
