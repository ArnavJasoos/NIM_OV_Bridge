#!/usr/bin/env python3
"""
NIM-OV-Bridge v3: Automated NIM Container → Intel iGPU Bridge
Ubuntu 22.04 LTS | i9-14900HX Intel UHD Graphics (Xe-LP)

Usage:
    python3 nim_ov_bridge.py \
        --container-url nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
        --ngc-api-key $NGC_API_KEY \
        --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --port 8000 \
        --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct
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
    env = os.environ.copy()
    env["NGC_API_KEY"] = api_key
    run(["docker", "login", registry, "-u", "$oauthtoken", "--password-stdin"],
        env=env, check=False)


def pull_or_inspect_image(container_url: str, existing: bool = False) -> str:
    if existing:
        print(f"[IMAGE] Using existing image: {container_url}")
        result = subprocess.run(["docker", "images", "-q", container_url],
                              capture_output=True, text=True, check=True)
        image_id = result.stdout.strip()
        if not image_id:
            print(f"[ERROR] Image {container_url} not found locally.")
            sys.exit(1)
        return image_id
    else:
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


def convert_hf_to_openvino(hf_model_id: str, model_name: str, quantize: str = "int8") -> Path:
    model_out = MODELS_DIR / model_name / "openvino"
    if (model_out / "openvino_model.xml").exists():
        print(f"[MODEL] OpenVINO IR already exists at {model_out}")
        return model_out

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Converting {hf_model_id} to OpenVINO IR ({quantize})...")
    print("[MODEL] This may take 10-30 minutes and requires ~20GB RAM...")

    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", hf_model_id,
        "--task", "text-generation-with-past",
        "--weight-format", quantize,
        str(model_out)
    ]
    try:
        run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] optimum-cli failed ({e}), falling back to Python export...")
        fallback_export(hf_model_id, model_out, quantize)

    print(f"[OK] Model exported to: {model_out}")
    return model_out


def fallback_export(hf_model_id: str, output_dir: Path, quantize: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
import sys
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
print("Export complete.")
"""
    script_path = output_dir / "_export.py"
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
    parser.add_argument("--container-url", required=True)
    parser.add_argument("--ngc-api-key", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hf-model-id", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quantize", default="int8", choices=["fp16", "int8", "int4"])
    parser.add_argument("--use-existing-image", action="store_true")
    parser.add_argument("--skip-conversion", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("  NIM-OV-Bridge v3: Intel iGPU Deployment Tool")
    print("  Target: Ubuntu 22.04 LTS + i9-14900HX UHD Graphics")
    print("=" * 60)

    container_cmd = check_prerequisites()
    ngc_login(args.container_url, args.ngc_api_key)
    image_id = pull_or_inspect_image(args.container_url, existing=args.use_existing_image)
    nim_info = inspect_nim_structure(image_id)

    if not args.skip_conversion:
        ov_model_path = convert_hf_to_openvino(args.hf_model_id, args.model_name, args.quantize)
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