#!/usr/bin/env bash
# Install a local venv on Prithvi (A40, CUDA 13 driver) and run the slice.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pytest -q
python scripts/run_slice.py --epochs 60 --n-per-preset 16 --device cuda --out output/prithvi
