# Model checkpoints

The checkpoints are downloaded from Hugging Face and are excluded from Git.
[`scripts/download_models.py`](scripts/download_models.py) pins the revisions
below, stores the large files under `models/`, and keeps the BiRefNet cache
under `.hf-cache/`.

| Stage | Hugging Face repository | Pinned revision | Approximate size |
| --- | --- | --- | ---: |
| Direct3D-V3 shape denoisers | [`wushuang98/Direct3D-V3`](https://huggingface.co/wushuang98/Direct3D-V3/tree/78925d8647eb6f1b5ea75a27963bad6e1528f4f4) | `78925d8647eb6f1b5ea75a27963bad6e1528f4f4` | 15.3 GB |
| DINOv3 ViT-H+ image encoder | [`bschooled/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/bschooled/dinov3-vith16plus-pretrain-lvd1689m/tree/13ba4c856a65d794825216e8e38d88e59c0210c4) | `13ba4c856a65d794825216e8e38d88e59c0210c4` | 3.36 GB |
| Hunyuan3D-2.1 shape model | [`tencent/Hunyuan3D-2.1`](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-dit-v2-1) | `0b94677654c57bb9a6b6845cd7b704ccf551d327` | 7.37 GB |
| BiRefNet foreground segmentation | [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | varies by revision |

The download script points Direct3D-V3's `image_encoder` entry to the pinned
DINOv3 mirror, so clean installations do not need access to Meta's gated
repository. The mirror includes the upstream model files and license; follow
the DINOv3 license and model terms provided there.

Hunyuan3D-2.1's shape model is published under Tencent's Hunyuan Community
License. Review its [model card](https://huggingface.co/tencent/Hunyuan3D-2.1)
and the vendored [upstream license](third_party/Hunyuan3D-2.1/LICENSE) before
use. Generated geometry and all checkpoint files remain local to the user.
