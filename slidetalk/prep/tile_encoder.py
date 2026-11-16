"""Frozen tile encoder: Owkin's Phikon (ViT-B/16, pretrained on 40M TCGA tiles, 768-d, open weights).

phikon-v2 (ViT-L, 1024-d) is a drop-in, stronger option: --tile-encoder phikon-v2.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T

ENCODERS = {
    # name: (dim, input_px)
    "phikon": (768, 224),
    "phikon-v2": (1024, 224),
}


def build_tile_encoder(name: str, device: str):
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dim, px = ENCODERS[name]
    transform = T.Compose([T.Resize(px, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])

    if name in ("phikon", "phikon-v2"):
        from transformers import AutoModel
        repo = "owkin/phikon" if name == "phikon" else "owkin/phikon-v2"
        model = AutoModel.from_pretrained(repo, attn_implementation="eager").eval().half().to(device)

        def fwd(x):
            return model(pixel_values=x.half()).last_hidden_state[:, 0]
    else:
        raise ValueError(name)

    @torch.inference_mode()
    def encode(x: torch.Tensor) -> torch.Tensor:
        # explicit fp16 weights: MPS autocast is flaky on macOS 14
        return fwd(x.to(device, non_blocking=True)).float()

    return encode, transform, dim
