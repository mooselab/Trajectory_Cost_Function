#!/usr/bin/env bash
# One-time setup: uv, Python venv with vLLM + measurement deps, model download.
# Run from this folder on the server. No sudo needed.
set -euo pipefail
cd "$(dirname "$0")"
source config.sh

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$HF_HOME" logs data results
# uv-managed Python ships Python.h; the system python3.12 has no -dev headers (no sudo to add
# them), and Triton must compile a C launcher against them when vLLM starts.
uv venv .venv --python 3.12 --managed-python --clear
uv pip install --python .venv/bin/python vllm nvidia-ml-py aiohttp pandas numpy scipy matplotlib
uv pip freeze --python .venv/bin/python > env_freeze.txt

echo "downloading $MODEL to $HF_HOME (tens of GB)..."
.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

.venv/bin/python -c "import os, sysconfig; h = os.path.join(sysconfig.get_paths()['include'], 'Python.h'); assert os.path.exists(h), 'missing ' + h; print('Python.h ok')"
.venv/bin/python -c "import vllm, pynvml; print('vllm', vllm.__version__)"
echo "setup done"
