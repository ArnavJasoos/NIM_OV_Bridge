#!/bin/bash
set -e

echo "[SETUP] NIM-OV-Bridge Host Preparation for Ubuntu 22.04"
echo "[SETUP] Target: Intel i9-14900HX UHD Graphics"

apt-get update

echo "[SETUP] Installing Intel GPU compute drivers..."
apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    ocl-icd-libopencl1 \
    clinfo \
    libze-intel-gpu1 \
    libze-loader \
    libigdgmm12 \
    vainfo \
    intel-media-va-driver-non-free

echo "[SETUP] Verifying Intel GPU..."
clinfo -l || true

if lsmod | grep -q i915; then
    echo "[OK] i915 kernel module loaded"
else
    echo "[WARN] i915 module not loaded. Attempting to load..."
    modprobe i915 || echo "[WARN] Failed to load i915. Reboot may be required."
fi

if ! command -v docker &> /dev/null; then
    echo "[SETUP] Installing Docker..."
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    usermod -aG docker $SUDO_USER || true
    echo "[OK] Docker installed. Log out and back in for group changes."
fi

echo "[SETUP] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install optimum-intel[openvino] \
    openvino \
    openvino-genai \
    transformers \
    accelerate \
    huggingface-hub \
    nncf \
    fastapi \
    uvicorn \
    pydantic

echo "[SETUP] Verifying OpenVINO GPU plugin..."
python3 -c "import openvino as ov; core = ov.Core(); print('Available devices:', core.available_devices)" || true

echo ""
echo "========================================"
echo "  Host setup complete!"
echo "  Reboot recommended if i915 was updated."
echo "========================================"