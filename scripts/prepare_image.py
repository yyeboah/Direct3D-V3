#!/usr/bin/env python3
"""Create a transparent foreground image shared by both 3D stages."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direct3d.utils.image import BackgroundRemover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Image preparation requires CUDA; no CUDA device is available.")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    with Image.open(args.image) as source:
        image = source.convert("RGBA")
    if image.getchannel("A").getextrema() == (255, 255):
        image = BackgroundRemover()(image, "cuda")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Transparent foreground image saved to {args.output}")


if __name__ == "__main__":
    main()
