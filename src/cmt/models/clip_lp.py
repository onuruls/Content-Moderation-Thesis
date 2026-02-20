from __future__ import annotations
from typing import List
import torch
import open_clip
import numpy as np
from PIL import Image
import os

from cmt.utils.torch_utils import get_device, dtype_from_str, get_amp_context, l2_norm
from cmt.utils.image_utils import ensure_rgb
from cmt.models.base import Model

class CLIPLinearProbe(Model):
    def __init__(self, 
                 model_id: str = "ViT-L-14", 
                 pretrained: str = "openai", 
                 head_path: str | None = None,
                 l2norm: bool = True,
                 dtype: str = "float16"):
        self.device = get_device()
        self.dtype = dtype_from_str(dtype)
        self.l2norm = l2norm
        
        # Verify BF16 support
        if self.device == "cuda" and self.dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                self.dtype = torch.float16

        print(f"Loading CLIP Probe {model_id} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if head_path is None:
            raise ValueError("head_path must be provided for CLIPLinearProbe")
            
        print(f"Loading head from {head_path}...")
        if head_path.endswith(".pt") or head_path.endswith(".pth"):
            state = torch.load(head_path, map_location="cpu")
            if "w" in state:
                w = state["w"].float()
                b = state["b"].float()
            else:
                # Handle potential other formats if needed, or fail
                 w = state["fc.weight"].float().t()
                 b = state["fc.bias"].float()
        else:
            # Assume .npz
            npz = np.load(head_path)
            w = torch.from_numpy(npz["w"]).float()
            b = torch.from_numpy(npz["b"]).float()

        self.w = w.to(self.device).contiguous()
        self.b = b.to(self.device).contiguous()

    @torch.inference_mode()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        # Preprocess images
        tensors = []
        for img in images:
            tensors.append(self.preprocess(ensure_rgb(img)))
        
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        
        with get_amp_context(True, self.dtype):
            features = self.model.encode_image(batch)
            
        f = features.float()
        return l2_norm(f) if self.l2norm else f

    @torch.inference_mode()
    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        features = self.encode(images)
        # z = f @ w + b
        return (features @ self.w + self.b).squeeze(1)

    @torch.inference_mode()
    def prob(self, images: List[Image.Image]) -> np.ndarray:
        z = self.logits(images)
        z = torch.clamp(z, -50, 50)
        p = torch.sigmoid(z)
        return p.cpu().float().numpy()
