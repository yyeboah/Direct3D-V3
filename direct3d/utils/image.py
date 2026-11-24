import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class BackgroundRemover:
    """Foreground segmentation used for every reference image."""

    def __init__(self, local_files_only=False):
        from transformers import AutoModelForImageSegmentation

        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image, device):
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            with Image.open(image) as source:
                rgb = source.convert("RGB")
        batch = self.transform(rgb).unsqueeze(0).to(device)
        self.model.to(device)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                try:
                    predictions = self.model(batch)
                except RuntimeError as error:
                    if "GET was unable to find an engine" not in str(error):
                        raise
                    # Some Jetson cuDNN builds have no engine for BiRefNet's
                    # FP16 convolutions; PyTorch's CUDA fallback kernels work.
                    with torch.backends.cudnn.flags(enabled=False):
                        predictions = self.model(batch)
                alpha = predictions[-1].sigmoid()[0, 0].cpu()
        finally:
            self.model.cpu()
        mask = transforms.ToPILImage()(alpha).resize(rgb.size)
        rgb.putalpha(mask)
        return rgb


def prepare_image(image):
    """Composite an RGBA cutout into the tensor expected by the image encoder."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.copy()
    if image.mode != "RGBA":
        raise ValueError("A background-removed image must be RGBA.")
    rgba = np.asarray(image)
    foreground = rgba[:, :, 3] > 0
    if foreground.mean() >= 0.001:
        rows, cols = np.where(foreground)
        rgba = rgba[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    alpha = rgba[:, :, 3:] / 255.0
    rgb = (127 * (1 - alpha) + rgba[:, :, :3] * alpha).astype(np.uint8)
    # Preserve the channel order used after foreground compositing in the reference.
    image = Image.fromarray(rgb[:, :, ::-1])
    width, height = image.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    transform = transforms.Compose(
        [
            transforms.Pad((left, top, side - width - left, side - height - top), fill=255),
            transforms.Resize((1536, 1536)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)
