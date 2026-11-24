#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

IMAGE="${1:-assets/input_imgs/6.png}"
OUTPUT="${2:-generated-e2e.glb}"
case "$OUTPUT" in
  *.glb) COARSE="${OUTPUT%.glb}.coarse.obj" ;;
  *) COARSE="${OUTPUT}.coarse.obj" ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Initialize Conda, then run this script again." >&2
  exit 127
fi
for env_dir in "$ROOT/.conda-envs/hunyuan3d-v21" "$ROOT/.conda-envs/direct3d-v3"; do
  if [[ ! -x "$env_dir/bin/python" ]]; then
    echo "Environment is missing: $env_dir" >&2
    echo "Set up Hunyuan with ./scripts/setup_hunyuan_env.sh and restore the Direct3D-V3 environment." >&2
    exit 2
  fi
done
if [[ ! -f "$IMAGE" ]]; then
  echo "Input image not found: $IMAGE" >&2
  exit 2
fi
case "${IMAGE,,}" in
  *.png|*.jpg|*.jpeg|*.webp) ;;
  *) echo "Use an input image with a .png, .jpg, .jpeg or .webp extension." >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export HY3DGEN_MODELS="$ROOT/models"
if [[ ! -f "$ROOT/models/Direct3D-V3/shape_transformers_1024/model.safetensors" || \
      ! -f "$ROOT/models/Direct3D-V3/image_encoder/model.safetensors" || \
      ! -f "$ROOT/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt" || \
      ! -d "$HF_HOME/hub/models--ZhengPeng7--BiRefNet" ]]; then
  echo "Fetching pinned model checkpoints from Hugging Face (first run only)..."
  conda run --no-capture-output -p "$ROOT/.conda-envs/direct3d-v3" \
    python "$ROOT/scripts/download_models.py"
fi
mkdir -p "$(dirname -- "$OUTPUT")" "$(dirname -- "$COARSE")"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
PREPARED_IMAGE="$WORK_DIR/foreground.png"

echo "[1/3] Preparing a transparent foreground image..."
conda run --no-capture-output -p "$ROOT/.conda-envs/direct3d-v3" \
  python "$ROOT/scripts/prepare_image.py" --image "$IMAGE" --output "$PREPARED_IMAGE"

echo "[2/3] Generating coarse shape with Hunyuan3D-2.1..."
conda run --no-capture-output -p "$ROOT/.conda-envs/hunyuan3d-v21" \
  python "$ROOT/scripts/generate_coarse.py" --image "$PREPARED_IMAGE" --output "$COARSE"

echo "[3/3] Refining the shape with Direct3D-V3..."
conda run --no-capture-output -p "$ROOT/.conda-envs/direct3d-v3" \
  python "$ROOT/infer.py" --model "$ROOT/models/Direct3D-V3" \
    --image "$PREPARED_IMAGE" --background-removed \
    --coarse-mesh "$COARSE" --resolution 1024 --output "$OUTPUT"

echo "Finished: $OUTPUT"
