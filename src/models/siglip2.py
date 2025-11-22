# src/models/siglip2.py
import torch
from typing import List, Dict, Tuple
from transformers import AutoProcessor, SiglipModel
from src.eval.util import img_rgba_to_rgb, safe_norm, cosine_margin, get_logit_scale, dtype_from_str

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

class SigLIP2ZS:
	# --- init ---
	def __init__(self, model_id="google/siglip2-so400m-patch14-384", dtype="float16"):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.dtype = dtype_from_str(dtype)
		self.model = SiglipModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
		for p in self.model.parameters(): p.requires_grad_(False)
		self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
		self._txt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
		self._gen_neg_proto: torch.Tensor | None = None
		self.tau = torch.tensor(1.0, device=self.device)

	# --- text encoding ---
	def _encode_texts(self, texts: List[str]) -> torch.Tensor:
		texts = [t.lower() for t in texts]
		enc = self.processor(text=texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
		enc = {k: v.to(self.device) for k, v in enc.items()}
		with torch.inference_mode():
			txt = self.model.get_text_features(
				input_ids=enc["input_ids"],
				attention_mask=enc.get("attention_mask", None),
			)
		return safe_norm(txt.float())

	# --- prototypes ---
	def _get_prototypes(self, category: str) -> Tuple[torch.Tensor, torch.Tensor]:
		key = (category or "unsafe").lower().strip()
		if key in self._txt_cache:
			return self._txt_cache[key]
		pos_texts = [t.format(cat=key) for t in POS_TEMPLATES]
		pos_proto = safe_norm(self._encode_texts(pos_texts).mean(dim=0, keepdim=True))
		if self._gen_neg_proto is None:
			self._gen_neg_proto = safe_norm(self._encode_texts(NEG_TEMPLATES).mean(dim=0, keepdim=True))
		neg_proto = self._gen_neg_proto
		self._txt_cache[key] = (pos_proto, neg_proto)
		return pos_proto, neg_proto

	# --- public API ---
	def predict_proba(self, pil_img, category):
		pil_img = img_rgba_to_rgb(pil_img)
		enc = self.processor(images=pil_img, return_tensors="pt")
		enc = {k: v.to(self.device) for k, v in enc.items()}
		enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
		with torch.inference_mode():
			img_f = self.model.get_image_features(pixel_values=enc["pixel_values"])
			img_f = safe_norm(img_f.float())
			pos_proto, neg_proto = self._get_prototypes(category)
			margin = cosine_margin(img_f, pos_proto, neg_proto)
			scale = get_logit_scale(self.model)
			p = torch.sigmoid(scale * margin)
		return float(p.item())

	@torch.inference_mode()
	def predict_proba_batch(self, pil_list, categories=None):
		enc = self.processor(images=[img_rgba_to_rgb(im) for im in pil_list], return_tensors="pt")
		enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
		enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
		img_f = self.model.get_image_features(pixel_values=enc["pixel_values"])
		img_f = safe_norm(img_f.float())
		pos_list, neg_list = [], []
		for cat in categories:
			pp, np_ = self._get_prototypes(cat)
			pos_list.append(pp)
			neg_list.append(np_)
		pos_stack = torch.cat(pos_list, dim=0)
		neg_stack = torch.cat(neg_list, dim=0)
		scale = get_logit_scale(self.model)
		margin = cosine_margin(img_f, pos_stack, neg_stack)
		p = torch.sigmoid(scale * margin).reshape(-1)
		return p.cpu().tolist()
