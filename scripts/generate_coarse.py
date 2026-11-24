#!/usr/bin/env python3
"""Generate the coarse Hunyuan3D shape used by Direct3D-V3 refinement."""

import argparse
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=50, type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    shape_repo = root / "third_party/Hunyuan3D-2.1/hy3dshape"
    sys.path.insert(0, str(shape_repo))
    os.environ.setdefault("HY3DGEN_MODELS", str(root / "models"))

    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D requires CUDA; no CUDA device is available.")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16
    )
    meshes = pipeline(image=str(args.image), num_inference_steps=args.steps)
    if not meshes:
        raise RuntimeError("Hunyuan3D returned no mesh.")
    mesh = meshes[0]
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError("Hunyuan3D could not extract a usable surface from the generated shape.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.output))
    print(f"Hunyuan3D coarse mesh saved to {args.output}")


if __name__ == "__main__":
    main()
