#!/usr/bin/env python3
"""Fetch the pinned inference checkpoints into ignored local model/cache dirs."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

from huggingface_hub import snapshot_download


DIRECT3D_REPO = "wushuang98/Direct3D-V3"
DIRECT3D_REVISION = "78925d8647eb6f1b5ea75a27963bad6e1528f4f4"
DINO_REPO = "bschooled/dinov3-vith16plus-pretrain-lvd1689m"
DINO_REVISION = "13ba4c856a65d794825216e8e38d88e59c0210c4"
HUNYUAN_REPO = "tencent/Hunyuan3D-2.1"
HUNYUAN_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def fetch(repo_id: str, revision: str, **kwargs) -> str:
    print(f"Downloading {repo_id}@{revision}", flush=True)
    return snapshot_download(repo_id=repo_id, revision=revision, **kwargs)


def main() -> None:
    models = ROOT / "models"
    direct3d_dir = models / "Direct3D-V3"
    fetch(DIRECT3D_REPO, DIRECT3D_REVISION, local_dir=direct3d_dir)

    image_encoder_dir = direct3d_dir / "image_encoder"
    fetch(DINO_REPO, DINO_REVISION, local_dir=image_encoder_dir)
    index_path = direct3d_dir / "model_index.json"
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    model_index["image_encoder"] = "./image_encoder"
    index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    hunyuan_dir = models / "tencent" / "Hunyuan3D-2.1"
    fetch(
        HUNYUAN_REPO,
        HUNYUAN_REVISION,
        local_dir=hunyuan_dir,
        allow_patterns=["hunyuan3d-dit-v2-1/*"],
    )

    hf_home = Path(os.environ["HF_HOME"])
    fetch(
        BIREFNET_REPO,
        BIREFNET_REVISION,
        cache_dir=hf_home / "hub",
    )
    print(f"Models are ready under {models}; Hugging Face cache: {hf_home}")


if __name__ == "__main__":
    main()
