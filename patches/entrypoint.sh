#!/bin/bash
set -e

echo "[NIM-OV-BRIDGE] Starting Intel iGPU Inference Bridge..."
echo "[NIM-OV-BRIDGE] Model path: ${MODEL_PATH}"
echo "[NIM-OV-BRIDGE] Device: ${OPENVINO_DEVICE:-GPU}"
echo "[NIM-OV-BRIDGE] API Port: ${API_PORT:-8000}"

if [ "${OPENVINO_DEVICE}" = "GPU" ] || [ -z "${OPENVINO_DEVICE}" ]; then
    echo "[NIM-OV-BRIDGE] Checking Intel GPU runtime..."
    if command -v clinfo &> /dev/null; then
        clinfo -l 2>/dev/null | grep -i "Intel" || echo "[WARN] No Intel GPU found in clinfo"
    fi
    if [ -d /dev/dri ]; then
        ls -la /dev/dri/ || true
    else
        echo "[WARN] /dev/dri not found. GPU passthrough may not be configured."
    fi
fi

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/intel/openvino/runtime/lib:/usr/lib/intel:${LD_LIBRARY_PATH}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[FATAL] Model path ${MODEL_PATH} not mounted."
    exit 1
fi

if [ ! -f "${MODEL_PATH}/openvino_model.xml" ] && [ ! -f "${MODEL_PATH}/openvino_model.bin" ]; then
    echo "[WARN] Expected OpenVINO IR files not found in ${MODEL_PATH}"
    ls -la "${MODEL_PATH}" || true
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "[NIM-OV-BRIDGE] Launching API server on port ${API_PORT:-8000}..."
cd /opt/nim-ov-bridge
exec python3 /opt/nim-ov-bridge/openvino_api_server.py