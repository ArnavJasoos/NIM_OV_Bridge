#!/bin/bash
set -e

echo "[SETUP] NIM-OV-Bridge Host Preparation for Ubuntu 22.04"
echo "[SETUP] Target: Intel i9-14900HX UHD Graphics (Raptor Lake / Xe-LP)"

# Clean up or configure Docker repository first to avoid apt-get update conflicts
if grep -r "download.docker.com" /etc/apt/sources.list.d/ --exclude="docker.list" &>/dev/null; then
    echo "[SETUP] Docker repository already configured in another file (e.g., docker.sources). Removing temporary docker.list if present..."
    rm -f /etc/apt/sources.list.d/docker.list
else
    echo "[SETUP] Configuring Docker repository (amd64 only)..."
    install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.asc ]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc
    fi
    # Write Docker source with explicit amd64 architecture only
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
fi

# Fix 2: Add Intel GPU repository for Level Zero drivers
echo "[SETUP] Adding Intel GPU repository..."
if [ ! -f /etc/apt/keyrings/intel-graphics.gpg ]; then
    curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor -o /etc/apt/keyrings/intel-graphics.gpg
fi

# Intel GPU repository for Jammy (Ubuntu 22.04)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy unified" > /etc/apt/sources.list.d/intel-graphics.list

apt-get update

echo "[SETUP] Installing Intel GPU compute drivers..."
apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    ocl-icd-libopencl1 \
    clinfo \
    libze-intel-gpu1 \
    libze1 \
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
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    usermod -aG docker $SUDO_USER || true
    echo "[OK] Docker installed. Log out and back in for group changes."
fi

echo "[SETUP] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install --ignore-installed optimum-intel[openvino] \
    optimum \
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
