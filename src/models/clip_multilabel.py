from __future__ import annotations
from typing import List
import torch
import open_clip
from PIL import Image
from contextlib import nullcontext

def _dtype_from_str(s: str | None) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):   return torch.float16
    if s in ("bf16", "bfloat16"):          return torch.bfloat16
    return torch.float32

def _amp(enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast("cuda", dtype=dtype) if enabled else nullcontext()

def _l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

class CLIPMultiLabelTrained:
    def __init__(self, head_path: str, dtype: str = "float16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # AMP dtype: bf16 destekliyse bf16, aksi halde fp16
        want = _dtype_from_str(dtype)
        if self.device == "cuda" and want == torch.bfloat16 and not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            want = torch.float16
        self.param_dtype = want
        self.use_amp = (self.device == "cuda")

        # CLIP backbone + preprocess
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=self.device
        )
        self.model.eval().requires_grad_(False)

        # Perf bayrakları (ZS/LP ile aynı)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Trained head yükle (fine-tune formatı: fc.weight/fc.bias)
        ckpt  = torch.load(head_path, map_location="cpu")
        state = ckpt["state_dict"] or ckpt["model_state"]
        class_names = [str(x).lower() for x in ckpt["class_names"]]

        # fc or head
        w = state["fc.weight"].to(self.device).float().t()  # [D,C]
        b = state["fc.bias"].to(self.device).float()        # [C]
        self.w, self.b = w.contiguous(), b.contiguous()

        # "unsafe" raporu için nudity index (eğitimle aynı seçim)
        self.i_nudity = class_names.index("nudity")
        self.i_sexual = class_names.index("sexy")  # (gerekirse max için hazır)

    # ------ encode ------
    @torch.inference_mode()
    def _encode(self, pil_list: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_list], 0)
        # aynı bellek bayrakları
        x = x.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        with _amp(self.use_amp, self.param_dtype):
            f = self.model.encode_image(x)  # [B, D] (fp16/bf16 compute)
        return _l2norm_rows(f.float())     # başlık fp32’de

    # ------ public API ------
    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List[Image.Image], categories=None) -> List[float]:
        f = self._encode(pil_list)               # [B, D]
        z = f @ self.w + self.b                  # [B, C]
        z = torch.clamp(z, -50, 50)
        p = torch.sigmoid(z)                     # [B, C]
        unsafe = p[:, self.i_nudity]             # raporlama: nudity kolonu
        return unsafe.float().cpu().tolist()

    @torch.inference_mode()
    def predict_proba(self, pil_img, category: str | None = None) -> float:
        return self.predict_proba_batch([pil_img])[0]
