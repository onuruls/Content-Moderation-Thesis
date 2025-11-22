# src/models/clip.py
import os
import torch
from typing import List, Dict, Tuple
import open_clip
from contextlib import nullcontext

from src.eval.util import img_rgba_to_rgb, safe_norm, cosine_margin, get_logit_scale

POS_TEMPLATES = [
    "a photo containing {cat} related unsafe content",
    "an image showing {cat} unsafe content",
    "a depiction of {cat} unsafe content",
]
NEG_TEMPLATES = [
    "a harmless, safe image",
    "an image with no unsafe content",
    "a benign photo",
]

def _dtype_from_env() -> torch.dtype:
    s = os.getenv("CLIP_DTYPE", "float16").lower()
    if s in ("bf16","bfloat16"): return torch.bfloat16
    if s in ("fp16","float16","half"): return torch.float16
    return torch.float32

class CLIPZS:
    def __init__(self, name="ViT-L-14", pretrained="openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(name)
        self._txt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._gen_neg_proto: torch.Tensor | None = None
        self.temperature = float(os.getenv("CLIP_TEMPERATURE", "1.0"))

        # --- perf knobs (trained sürümle aynı) ---
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.use_amp = (self.device == "cuda")
        self.amp_dtype = _dtype_from_env()

        with torch.inference_mode():
            # open_clip modellerinde logit_scale genelde learnable; exp() ile skala elde edilir.
            ls = getattr(self.model, "logit_scale", None)
            if ls is None:
                self._scale_cached = None
            else:
                # tensör olarak sakla; .item() YOK (senkronizasyonu engelle)
                self._scale_cached = ls.exp().detach()

    # ---- text encoding ----
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts).to(self.device, non_blocking=True)
        with torch.inference_mode():
            txt = self.model.encode_text(toks).float()
        return safe_norm(txt)

    # ---- prototypes ----
    def _get_prototypes(self, category: str | None) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (category or "unsafe").lower()
        if key in self._txt_cache:
            return self._txt_cache[key]
        pos_texts = [t.format(cat=key) for t in POS_TEMPLATES]
        pos_proto = safe_norm(self._encode_texts(pos_texts).mean(dim=0, keepdim=True))
        if self._gen_neg_proto is None:
            self._gen_neg_proto = safe_norm(self._encode_texts(NEG_TEMPLATES).mean(dim=0, keepdim=True))
        neg_proto = self._gen_neg_proto
        self._txt_cache[key] = (pos_proto, neg_proto)
        return pos_proto, neg_proto

    def _get_scale(self) -> torch.Tensor:
        # Gerekirse tekrar oku ama yine tensör olarak dön.
        if self._scale_cached is not None:
            return self._scale_cached
        ls = getattr(self.model, "logit_scale", None)
        return ls.exp().detach() if ls is not None else torch.tensor(1.0, device=self.device)

    # ---- public API ----
    @torch.inference_mode()
    def predict_proba(self, pil_img, category):
        pil_img = img_rgba_to_rgb(pil_img)
        img = self.preprocess(pil_img).unsqueeze(0).to(self.device, non_blocking=True, memory_format=torch.channels_last)
        ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        with ctx:
            img_f = self.model.encode_image(img).float()
        img_f = safe_norm(img_f)
        pos_proto, neg_proto = self._get_prototypes(category)
        margin = cosine_margin(img_f, pos_proto, neg_proto)  # -> [1,1] tensör
        scale = self._get_scale()  # -> cihaz tensörü
        logit = (scale * margin) / self.temperature  # hepsi GPU’da
        p = torch.sigmoid(logit)
        return float(p.item())

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List, categories=None):
        xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
        x = torch.stack(xs, dim=0).to(self.device, non_blocking=True, memory_format=torch.channels_last)
        ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        with ctx:
            img_f = self.model.encode_image(x).float()
        img_f = safe_norm(img_f)
        scale = self._get_scale()

        if categories is None:
            pos_proto, neg_proto = self._get_prototypes(None)
            margin = cosine_margin(img_f, pos_proto, neg_proto)  # [B,1]
            p = torch.sigmoid((scale * margin) / self.temperature).squeeze(1)
            return p.cpu().tolist()

        # (az kullanılıyor ama gerekliyse)
        res: List[float] = []
        for i, cat in enumerate(categories):
            pos_proto, neg_proto = self._get_prototypes(cat)
            m = cosine_margin(img_f[i:i+1], pos_proto, neg_proto)
            p = torch.sigmoid((scale * m) / self.temperature)
            res.append(float(p.item()))
        return res
