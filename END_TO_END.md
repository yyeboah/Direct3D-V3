# Image to refined 3D mesh

## One-time setup

Use a Linux NVIDIA GPU machine with Conda. The project environment must be at
`.conda-envs/direct3d-v3`, use Python 3.10, and have a CUDA-enabled PyTorch and
torchvision build matched to that machine's driver/toolkit. PyTorch installation
is hardware-specific; on Jetson, use a JetPack-compatible NVIDIA/PyTorch build
instead of a CPU-only PyPI wheel.

Create that environment, install the matching CUDA PyTorch build, then install
the project dependencies and CUDA extensions using the repository's installation
instructions. For Jetson, both CUDA extensions and FlashAttention must be built
for the installed PyTorch/CUDA/architecture combination:

```bash
conda create -p "$PWD/.conda-envs/direct3d-v3" python=3.10 -y
# Install CUDA-enabled torch and torchvision for your platform in this environment.
conda run -p "$PWD/.conda-envs/direct3d-v3" python -m pip install -r requirements.txt
conda run -p "$PWD/.conda-envs/direct3d-v3" python -m pip install --no-build-isolation ./third_party/voxelization ./third_party/fill_holes
```

Install a FlashAttention 2 build compatible with that PyTorch before running
inference. The environment used to validate this project was Python 3.10 with
PyTorch 2.10.0+CUDA 12.6 on JetPack 6.2.1.

Create the separate Hunyuan environment from that CUDA-enabled PyTorch stack:

```bash
./scripts/setup_hunyuan_env.sh
```

This copies the PyTorch and CUDA packages from `.conda-envs/direct3d-v3` into a
Python 3.10 environment and installs the pinned Hunyuan dependencies. The first
runner launch automatically downloads the pinned Direct3D-V3, DINOv3, Hunyuan,
and BiRefNet checkpoints listed in [MODEL_SOURCES.md](MODEL_SOURCES.md). This
needs roughly 27 GB of checkpoint storage plus working space and an internet
connection. To prefetch them before running inference:

```bash
conda run -p "$PWD/.conda-envs/direct3d-v3" python scripts/download_models.py
```

`runner.sh` prepares a transparent foreground, generates a coarse mesh, and
refines it in one command:

1. Direct3D-V3's BiRefNet removes the background if the input does not already
   have transparency.
2. Hunyuan3D-2.1 generates a coarse shape from the prepared image.
3. Direct3D-V3 refines that shape using the same prepared image.

From the repository root, run:

```bash
./runner.sh path/to/image.png path/to/refined.glb
```

Both arguments are optional. With no arguments, the runner reads
`assets/input_imgs/6.png` and writes `generated-e2e.glb`. It also keeps the
intermediate Hunyuan mesh as `generated-e2e.coarse.obj`. The runner expects the
two project environments at `.conda-envs/hunyuan3d-v21` and
`.conda-envs/direct3d-v3`; the latter is the existing Direct3D-V3 setup.

Hunyuan3D-2.1 is licensed under Tencent's Hunyuan Community License. Review
the [upstream license](third_party/Hunyuan3D-2.1/LICENSE) for usage terms.
This runner generates shape geometry only; it does not run Hunyuan's optional
texture-generation stage.
