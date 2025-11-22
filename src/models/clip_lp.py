# src/models/clip_lp.py
from __future__ import annotations
import torch
import open_clip
from typing import List
from contextlib import nullcontext

def _dtype_from_str(s: str | None) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):   return torch.float16
    if s in ("bf16", "bfloat16"):          return torch.bfloat16
    return torch.float32

def _amp(enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast("cuda", dtype=dtype) if enabled else nullcontext()

def l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

class CLIPLinearProbe:
    def __init__(self,
                 name: str = "ViT-L-14",
                 pretrained: str = "openai",
                 head_path: str | None = None,
                 l2norm: bool = True,
                 use_amp: bool = True,
                 dtype: str = "float16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.param_dtype = _dtype_from_str(dtype)
        self.use_amp = bool(use_amp) and (self.device == "cuda")

        # CLIP backbone + preprocess
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)

        # perf flags (fine-tuned script ile aynı)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            try: torch.backends.cudnn.allow_tf32 = True
            except Exception: pass
            torch.backends.cudnn.benchmark = True
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass

        # Linear head yükleme: .npz (w[D,1], b[1]) veya .pt/.pth (w,b)
        if head_path is None:
            raise ValueError("head_path must be provided.")
        if head_path.endswith(".pt") or head_path.endswith(".pth"):
            state = torch.load(head_path, map_location="cpu")
            w = state["w"].float()   # [D,1]
            b = state["b"].float()   # [1]
        else:
            import numpy as np
            npz = np.load(head_path)
            w = torch.from_numpy(npz["w"]).float()   # [D,1]
            b = torch.from_numpy(npz["b"]).float()   # [1]

        self.w = w.to(self.device, non_blocking=True).contiguous()
        self.b = b.to(self.device, non_blocking=True).contiguous()
        self.l2norm = bool(l2norm)

    @torch.inference_mode()
    def _encode(self, pil_list: List) -> torch.Tensor:
        # CPU -> GPU kopyasında pin_memory + channels_last
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_list], 0)
        if self.device == "cuda":
            x = x.pin_memory()
        x = x.to(self.device, non_blocking=True, memory_format=torch.channels_last)

        with _amp(self.use_amp, self.param_dtype):
            f = self.model.encode_image(x)  # [B, D] (AMP ise fp16/bf16 compute)
        f = f.float()  # linear head fp32

        return l2norm_rows(f) if self.l2norm else f

    # -------- Public API (değişmedi) --------
    @torch.inference_mode()
    def predict_logit_batch(self, pil_list: List, categories=None) -> list[float]:
        f = self._encode(pil_list)                # [B, D]
        logit = (f @ self.w + self.b).squeeze(1)  # [B]
        return logit.detach().cpu().tolist()

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List, categories=None) -> list[float]:
        f = self._encode(pil_list)                # [B, D]
        logit = (f @ self.w + self.b).squeeze(1)  # [B]
        logit = torch.clamp(logit, -50, 50)       # numeric stability
        p = torch.sigmoid(logit)
        return p.detach().cpu().tolist()

    @torch.inference_mode()
    def predict_proba(self, pil_img, _category=None) -> float:
        return self.predict_proba_batch([pil_img])[0]
