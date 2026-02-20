from __future__ import annotations
from typing import List
import torch
import open_clip
import numpy as np
from PIL import Image

from cmt.utils.torch_utils import get_device, dtype_from_str, get_amp_context, l2_norm
from cmt.utils.image_utils import ensure_rgb
from cmt.models.base import Model

class CLIPMultiLabelTrained(Model):
    def __init__(self, head_path: str, dtype: str = "float16", arch: str = "ViT-L-14", pretrained: str = "openai"):
        self.device = get_device()
        self.dtype = dtype_from_str(dtype)
        
        # Verify BF16 support
        if self.device == "cuda" and self.dtype == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print("Warning: BF16 not supported, falling back to FP16")
                self.dtype = torch.float16

        # Load Backbone
        print(f"Loading CLIP {arch} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        
        # Enable TF32 if on CUDA
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load Head
        print(f"Loading head from {head_path}...")
        ckpt = torch.load(head_path, map_location="cpu")
        state = ckpt.get("state_dict") or ckpt.get("model_state")
        self.class_names = [str(x).lower() for x in ckpt["class_names"]]
        
        # Head weights: shape [D, C] and bias [C]
        w = state["head.weight"].to(self.device).float().t()
        b = state["head.bias"].to(self.device).float()
        self.w, self.b = w.contiguous(), b.contiguous()
        
        self.logit_scale = torch.tensor(100.0, device=self.device)

    @torch.inference_mode()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        # Preprocess images
        tensors = []
        for img in images:
            tensors.append(self.preprocess(ensure_rgb(img)))
        
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        
        with get_amp_context(True, self.dtype):
            features = self.model.encode_image(batch)
            
        return l2_norm(features.float())

    @torch.inference_mode()
    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        features = self.encode(images)
        # z = f @ w + b
        return features @ self.w + self.b

    @torch.inference_mode()
    def prob(self, images: List[Image.Image]) -> np.ndarray:
        z = self.logits(images)
        # Clamp for stability
        z = torch.clamp(z, -50, 50)
        p = torch.sigmoid(z)
        return p.cpu().float().numpy()
