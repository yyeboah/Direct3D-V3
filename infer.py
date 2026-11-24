import argparse
from pathlib import Path

from direct3d import Direct3DV3Pipeline
from direct3d.pipeline import DEFAULT_MAX_TOKENS, DEFAULT_MODEL_ID, DEFAULT_TARGET_FACES, RESOLUTIONS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def input_pairs(image, coarse, output):
    if image.is_dir():
        if not coarse.is_dir():
            raise ValueError(
                "For an image directory, supply a coarse mesh directory with matching .obj stems."
            )
        images = sorted(p for p in image.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images found in {image}")
        pairs = [(p, coarse / f"{p.stem}.obj", output / f"{p.stem}.glb") for p in images]
    else:
        pairs = [(image, coarse, output)]
    outputs = set()
    for source, mesh, dest in pairs:
        if not source.is_file() or not mesh.is_file():
            raise FileNotFoundError(source if not source.is_file() else mesh)
        if dest.suffix.lower() not in {".obj", ".glb", ".ply"}:
            raise ValueError("Choose an OBJ, GLB or PLY output filename.")
        resolved = dest.resolve()
        if resolved in outputs or resolved in (mesh.resolve(), source.resolve()):
            raise ValueError(f"Conflicting output path: {dest}")
        outputs.add(resolved)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh from a reference image.")
    parser.add_argument(
        "--model", help=f"Hugging Face repository ID or local model directory (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Run a single stage from a local safetensors checkpoint"
    )
    parser.add_argument("--image", "--infer_path", type=Path, required=True, help="Input image or directory")
    parser.add_argument(
        "--coarse-mesh", type=Path, required=True, help="Coarse mesh or matching OBJ directory"
    )
    parser.add_argument("--output", type=Path, help="Output mesh or batch output directory")
    parser.add_argument("--resolution", "--voxel_size", type=int, choices=RESOLUTIONS, default=1024)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="One step count for all stages or one per active stage in resolution order",
    )
    parser.add_argument(
        "--cfg",
        "--guidance-scale",
        nargs="+",
        type=float,
        help="One CFG scale for all stages or one per active stage in resolution order",
    )
    for stage in RESOLUTIONS:
        parser.add_argument(f"--steps-{stage}", type=int, help=f"Override the step count for {stage} only")
        parser.add_argument(f"--cfg-{stage}", type=float, help=f"Override the CFG scale for {stage} only")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--background-removed", action="store_true",
        help="Skip segmentation when the input already has a transparent background",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum sparse tokens per stage (default: %(default)s). Lower this if you run out of GPU memory. "
             "The mesh may lose fine surface detail.",
    )
    parser.add_argument(
        "--mc-threshold", "--mc_threshold", type=float, help="Override the threshold for every stage"
    )
    clean = parser.add_mutually_exclusive_group()
    clean.add_argument(
        "--clean",
        action="store_const",
        const=DEFAULT_TARGET_FACES,
        dest="target_faces",
        help="Use the default final target of 2 million triangles",
    )
    clean.add_argument("--target-faces", type=int, help="Custom final triangle budget")
    clean.add_argument(
        "--no-simplify",
        action="store_const",
        const=None,
        dest="target_faces",
        help="Keep all final triangles",
    )
    parser.set_defaults(target_faces=DEFAULT_TARGET_FACES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stages = (args.resolution,) if args.checkpoint else tuple(r for r in RESOLUTIONS if r <= args.resolution)

    def stage_values(option):
        shared = getattr(args, option)
        values = {}
        if shared:
            if len(shared) == 1:
                values = dict.fromkeys(stages, shared[0])
            elif len(shared) == len(stages):
                values = dict(zip(stages, shared))
            else:
                parser.error(f"--{option} expects one value or one per active stage {stages}.")
        for stage in stages:
            override = getattr(args, f"{option}_{stage}")
            if override is not None:
                values[stage] = override
        return values

    steps = stage_values("steps")
    guidance = stage_values("cfg")
    output = args.output or Path("output" if args.image.is_dir() else "output.glb")
    try:
        pairs = input_pairs(args.image, args.coarse_mesh, output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.checkpoint is None:
        pipe = Direct3DV3Pipeline.from_pretrained(args.model or DEFAULT_MODEL_ID)
    else:
        pipe = Direct3DV3Pipeline.from_single_file(args.checkpoint, resolution=args.resolution, config_repo=args.model)
    pipe.to(args.device)
    for image, coarse, dest in pairs:
        result = pipe(
            image,
            coarse,
            resolution=args.resolution,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=args.seed,
            max_tokens=args.max_tokens,
            mc_threshold=args.mc_threshold,
            target_faces=args.target_faces,
            background_removed=args.background_removed,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.export(dest)
        print(f"{dest} ({len(result.faces):,} triangles)")


if __name__ == "__main__":
    main()
