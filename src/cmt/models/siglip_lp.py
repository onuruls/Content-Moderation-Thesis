from __future__ import annotations
from typing import List
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipModel

from cmt.utils.torch_utils import get_device, dtype_from_str, get_amp_context, l2_norm
from cmt.utils.image_utils import ensure_rgb
from cmt.models.base import Model

class SigLIPLinearProbe(Model):
    def __init__(self, 
                 model_id: str = "google/siglip-so400m-patch14-384",
                 head_path: str | None = None,
                 l2norm: bool = True,
                 dtype: str = "float16"):
        self.device = get_device()
        self.dtype = dtype_from_str(dtype)
        self.l2norm = l2norm
        
        print(f"Loading SigLIP Probe {model_id}...")
        self.model = SiglipModel.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

        if head_path is None:
            raise ValueError("head_path must be provided for SigLIPLinearProbe")
            
        print(f"Loading head from {head_path}...")
        if head_path.endswith(".pt") or head_path.endswith(".pth"):
            state = torch.load(head_path, map_location="cpu")
            w = state["w"].float()
            b = state["b"].float()
        else:
            npz = np.load(head_path)
            w = torch.from_numpy(npz["w"]).float()
            b = torch.from_numpy(npz["b"]).float()

        self.w = w.to(self.device).contiguous()
        self.b = b.to(self.device).contiguous()

    @torch.inference_mode()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        # SigLIP processor expects RGB
        imgs = [ensure_rgb(im) for im in images]
        inputs = self.processor(images=imgs, return_tensors="pt")
        
        input_pixels = inputs["pixel_values"].to(self.device)
        if self.dtype in (torch.float16, torch.bfloat16):
            input_pixels = input_pixels.to(self.dtype)
            
        with get_amp_context(True, self.dtype):
            features = self.model.get_image_features(pixel_values=input_pixels)
            
        f = features.float()
        return l2_norm(f) if self.l2norm else f

    @torch.inference_mode()
    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        features = self.encode(images)
        return (features @ self.w + self.b).squeeze(1)

    @torch.inference_mode()
    def prob(self, images: List[Image.Image]) -> np.ndarray:
        z = self.logits(images)
        z = torch.clamp(z, -50, 50)
        p = torch.sigmoid(z)
        return p.cpu().float().numpy()
