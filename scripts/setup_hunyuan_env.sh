#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="$ROOT/.conda-envs/hunyuan3d-v21"
SOURCE_ENV="${DIRECT3D_ENV:-$ROOT/.conda-envs/direct3d-v3}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 127
fi
if [[ -x "$ENV/bin/python" ]]; then
  conda run -p "$ENV" python -c 'import torch'
  conda run -p "$ENV" python -m pip install -r "$ROOT/requirements-hunyuan3d.txt"
  echo "Hunyuan3D environment updated. Run: $ROOT/runner.sh [input-image] [output.glb]"
  exit 0
fi
if [[ ! -d "$SOURCE_ENV/lib/python3.10/site-packages/torch" ]]; then
  echo "CUDA PyTorch was not found in $SOURCE_ENV; set DIRECT3D_ENV to its environment path." >&2
  exit 2
fi

conda create -y -p "$ENV" python=3.10
source_site="$SOURCE_ENV/lib/python3.10/site-packages"
target_site="$ENV/lib/python3.10/site-packages"
mkdir -p "$target_site"
for package in torch torch-*.dist-info torchvision torchvision-*.dist-info torchgen functorch nvidia nvidia_*.dist-info; do
  for item in "$source_site"/$package; do
    [[ -e "$item" ]] && cp -a "$item" "$target_site/"
  done
done

mkdir -p "$ENV/etc/conda/activate.d"
cat > "$ENV/etc/conda/activate.d/jetson-cuda.sh" <<'HOOK'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK

conda run -p "$ENV" python -m pip install -r "$ROOT/requirements-hunyuan3d.txt"
echo "Hunyuan3D environment ready. Run: $ROOT/runner.sh [input-image] [output.glb]"
