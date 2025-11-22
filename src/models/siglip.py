# src/models/siglip.py
from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel

from src.eval.util import img_rgba_to_rgb, safe_norm, cosine_margin, dtype_from_str

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

class SigLIPZS:
    def __init__(
        self,
        model_id: str = "google/siglip-so400m-patch14-384",
        dtype: str = "float16",
        local_files_only: bool = False,
        use_fast_tokenizer: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype_from_str(dtype)
        self.use_amp = (self.device == "cuda")

        # Model
        self.model = SiglipModel.from_pretrained(
            model_id, torch_dtype=self.dtype, local_files_only=local_files_only
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # >>> Değişiklik: AutoProcessor yerine ayrı bileşenler
        self.image_proc = AutoImageProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=use_fast_tokenizer, local_files_only=local_files_only
        )

        # logit_scale cache (GPU)
        _ls = getattr(self.model, "logit_scale", None)
        if _ls is not None:
            try:
                self._scale_cached = _ls.detach().exp().to(self.device, dtype=torch.float32)
            except Exception:
                self._scale_cached = torch.as_tensor(float(_ls.detach().exp().cpu()), device=self.device)
        else:
            self._scale_cached = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        self._txt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._gen_neg_proto: torch.Tensor | None = None

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    # ---- text encoding (GPU) ----
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype) if self.use_amp else torch.cuda.amp.autocast(enabled=False):
            txt = self.model.get_text_features(
                input_ids=tok["input_ids"],
                attention_mask=tok.get("attention_mask", None),
            )
        return safe_norm(txt.float())

    # ---- prototypes (GPU cache) ----
    def _get_prototypes(self, category: str | None) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (category or "unsafe").lower().strip()
        if key in self._txt_cache:
            return self._txt_cache[key]
        pos_texts = [t.format(cat=key) for t in POS_TEMPLATES]
        pos_proto = safe_norm(self._encode_texts(pos_texts).mean(dim=0, keepdim=True))
        if self._gen_neg_proto is None:
            self._gen_neg_proto = safe_norm(self._encode_texts(NEG_TEMPLATES).mean(dim=0, keepdim=True))
        pos_proto = pos_proto.to(self.device)
        neg_proto = self._gen_neg_proto.to(self.device)
        self._txt_cache[key] = (pos_proto, neg_proto)
        return pos_proto, neg_proto

    # ---- image encode (GPU) ----
    @torch.inference_mode()
    def _encode_images(self, pil_list: List) -> torch.Tensor:
        enc = self.image_proc(images=[img_rgba_to_rgb(im) for im in pil_list], return_tensors="pt")
        pix = enc["pixel_values"].to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=self.dtype) if self.use_amp else torch.cuda.amp.autocast(enabled=False):
            img_f = self.model.get_image_features(pixel_values=pix)
        return safe_norm(img_f.float())

    # ---- public API ----
    @torch.inference_mode()
    def predict_proba(self, pil_img, category: str):
        img_f = self._encode_images([pil_img])                      # [1,D]
        pos_proto, neg_proto = self._get_prototypes(category)       # [1,D]
        margin = cosine_margin(img_f, pos_proto, neg_proto)         # [1]
        p = torch.sigmoid(self._scale_cached * margin).reshape(-1)  # [1]
        return float(p[0].detach().cpu())

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List, categories: List[str] | None = None):
        img_f = self._encode_images(pil_list)                       # [B,D]
        scale = self._scale_cached
        if categories is None:
            pos_proto, neg_proto = self._get_prototypes(None)
            margin = cosine_margin(img_f, pos_proto, neg_proto)     # [B]
            p = torch.sigmoid(scale * margin).reshape(-1)           # [B]
            return p.detach().cpu().tolist()
        outs = []
        for i, cat in enumerate(categories):
            pos_proto, neg_proto = self._get_prototypes(cat)
            m = cosine_margin(img_f[i:i+1], pos_proto, neg_proto)   # [1]
            outs.append(torch.sigmoid(scale * m).reshape(-1))
        return torch.cat(outs, dim=0).detach().cpu().tolist()
