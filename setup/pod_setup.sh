#!/bin/bash
# pqm-lora's pod environment, one part per call. PQM's LoRA extension runs each part
# as an install step (G.5) in this checkout, and `serve` as its service.
#
#   gpu      the container really has a GPU (K.1.5's hosts that handed it none)
#   venv     Python 3.11 venv with torch 2.14.1 (CUDA 13.0) and requirements-pod.txt
#   weights  the tagger and JoyCaption at their pinned commits, into $HF_HOME
#   verify   torch sees the GPU, onnxruntime offers CUDA, the files are there
#   serve    the caption service on 127.0.0.1:$PQM_LORA_PORT (7878)
#
# Python and every package are pinned to what L.0's spike installed (2026-10-18);
# uv's --exclude-newer resolves what is not pinned as it stood that day.
set -u
part=${1:-}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${PQM_LORA_VENV:-/workspace/venvs/pqm-lora}
export HF_HOME=${HF_HOME:-/workspace/hf} UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uvcache}
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
PYV=3.11
TORCH="torch==2.14.1 torchvision==0.29.1"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
AS_OF=2026-10-18T00:00:00Z
fail() { echo "pod_setup $part FAILED: $1"; exit 3; }

case "$part" in
  gpu)
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail "nvidia-smi"
    python3 -c "import ctypes,sys; r=ctypes.CDLL('libcuda.so.1').cuInit(0); print('cuInit', r); sys.exit(r)" \
      || fail "the container has no usable GPU (cuInit failed; NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset})"
    ;;
  venv)
    command -v uv >/dev/null || pip install -q uv==0.13.1 || fail "installing uv"
    [ -x "$VENV/bin/python" ] || uv venv -q -p $PYV "$VENV" || fail "making the venv"
    uv pip install -q -p "$VENV/bin/python" $TORCH --index-url $TORCH_INDEX || fail "torch"
    uv pip install -q -p "$VENV/bin/python" --exclude-newer $AS_OF -r "$HERE/requirements-pod.txt" \
      || fail "requirements-pod.txt"
    "$VENV/bin/python" -c "import torch, onnxruntime, transformers; print('venv ok: torch', torch.__version__)" \
      || fail "the venv's imports"
    ;;
  weights)
    "$VENV/bin/python" -m pqm_lora.fetch || fail "fetching the models"
    ;;
  verify)
    "$VENV/bin/python" -m pqm_lora.verify || fail "the caption environment"
    ;;
  serve)
    exec "$VENV/bin/python" -m pqm_lora.serve --port "${PQM_LORA_PORT:-7878}"
    ;;
  *)
    echo "usage: pod_setup.sh gpu|venv|weights|verify|serve"; exit 2
    ;;
esac
