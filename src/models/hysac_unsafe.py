# src/models/hysac_unsafe.py
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTokenizer
from hysac.models import HySAC
from hysac import lorentz as L
from src.eval.util import dtype_from_str, img_rgba_to_rgb

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

def _to(x, device, dtype=None):
	if dtype is None:
		return x.to(device)
	return x.to(device=device, dtype=dtype)

def _hyp_sims(img_emb, txt_emb, curv):
	if img_emb.dim() == 1: img_emb = img_emb.unsqueeze(0)
	if txt_emb.dim() == 1: txt_emb = txt_emb.unsqueeze(0)
	d = L.pairwise_dist(img_emb, txt_emb, curv=curv)  # [n_img, n_txt]
	return -d

def _lse(x: torch.Tensor) -> torch.Tensor:
	m = torch.max(x)
	return m + torch.log(torch.clamp((x - m).exp().sum(), min=1e-12))


class HySAC_ZS:
	def __init__(self, model_id="aimagelab/HySAC", dtype="float16"):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.dtype = dtype_from_str(dtype)
		self.model = HySAC.from_pretrained(model_id, device=self.device).to(self.device).eval()
		self.image_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
		self.text_tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
		self._txt_cache = {}


	@torch.no_grad()
	def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
		return self.model.encode_image(pixel_values, True)

	@torch.no_grad()
	def _encode_text(self, texts: list[str]) -> torch.Tensor:
		enc = self.text_tok(
			texts,
			add_special_tokens=True,
			truncation=True,
			max_length=self.text_tok.model_max_length,
			padding=False,
			return_tensors=None,
		)
		ids_list = enc["input_ids"]
		tokens = [torch.tensor(ids, dtype=torch.long, device=self.device) for ids in ids_list]
		return self.model.encode_text(tokens, True)

	def predict_proba(self, pil_img, category: str) -> float:
		pil_img = img_rgba_to_rgb(pil_img)
		cat = (category or "unsafe").lower()
		pos_texts = [t.format(cat=cat) for t in POS_TEMPLATES]
		neg_texts = [t.format(cat=cat) for t in NEG_TEMPLATES]
		img = self.image_proc(images=pil_img, return_tensors="pt")["pixel_values"]
		img = _to(img, self.device, self.dtype)
		with torch.no_grad():
			img_emb = self._encode_image(img)
			pos_emb = self._encode_text(pos_texts)
			neg_emb = self._encode_text(neg_texts)
			curv = self.model.curv.exp()
			pos_sims = _hyp_sims(img_emb, pos_emb, curv=curv)  # shape [P]
			neg_sims = _hyp_sims(img_emb, neg_emb, curv=curv)  # shape [N]
			scale = self.model.logit_scale.exp()
			scale = torch.clamp(scale, max=torch.tensor(100.0, device=self.device))
			pos_logit = torch.logsumexp(scale * pos_sims, dim=1)  # [1]
			neg_logit = torch.logsumexp(scale * neg_sims, dim=1)  # [1]
			prob = torch.sigmoid((pos_logit - neg_logit)[0])      # scalar
		return float(prob.item())

	@torch.inference_mode()
	def predict_proba_batch(self, pil_list, categories=None):
		imgs = [img_rgba_to_rgb(im) for im in pil_list]
		px = self.image_proc(images=imgs, return_tensors="pt")["pixel_values"]
		px = _to(px, self.device, self.dtype)
		img_emb = self._encode_image(px)
		curv = self.model.curv.exp()
		cats = [(c or "unsafe").lower() for c in categories]
		idx_map = {}
		for i, c in enumerate(cats):
			idx_map.setdefault(c, []).append(i)
		scale = self.model.logit_scale.exp()
		scale = torch.clamp(scale, max=torch.tensor(100.0, device=self.device))
		out = torch.empty(len(cats), device=self.device, dtype=torch.float32)
		for c, idxs in idx_map.items():
			if c in self._txt_cache:
				pos_emb, neg_emb = self._txt_cache[c]
			else:
				pos_texts = [t.format(cat=c) for t in POS_TEMPLATES]
				neg_texts = [t.format(cat=c) for t in NEG_TEMPLATES]
				pos_emb = self._encode_text(pos_texts)
				neg_emb = self._encode_text(neg_texts)
				self._txt_cache[c] = (pos_emb, neg_emb)
			ie = img_emb[idxs]
			pos_sims = _hyp_sims(ie, pos_emb, curv)
			neg_sims = _hyp_sims(ie, neg_emb, curv)
			pos_logit = torch.logsumexp(scale * pos_sims, dim=1)
			neg_logit = torch.logsumexp(scale * neg_sims, dim=1)
			out[idxs] = torch.sigmoid(pos_logit - neg_logit)
		return out.cpu().tolist()

