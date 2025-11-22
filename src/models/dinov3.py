# src/models/dinov3.py
import torch
from torchvision import transforms
from typing import List, Dict, Tuple
from src.eval.util import dtype_from_str, img_rgba_to_rgb, safe_norm, cosine_margin, get_logit_scale

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

def _make_preprocess(img_size):
	return transforms.Compose([
		transforms.Resize((img_size, img_size), antialias=True),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.485, 0.456, 0.406),
		                     std=(0.229, 0.224, 0.225)),
	])

class DINOv3TXT_ZS:
	# --- init & config ---
	def __init__(self,
	             model_id="facebookresearch/dinov3",
	             model_entry="dinov3_vitl16_dinotxt_tet1280d20h24l",
	             dtype="float16"):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.image_size = 224
		self.dtype = dtype_from_str(dtype)

		hub_kwargs = {
			"weights": "C:/Users/onur-/Desktop/thesis/bench/src/models/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
			"backbone_weights": "C:/Users/onur-/Desktop/thesis/bench/src/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"}

		self.model, self.tokenizer = torch.hub.load(
			model_id, model_entry, source="github", trust_repo=True, **hub_kwargs
		)
		self.model = self.model.to(self.device).eval()
		for p in self.model.parameters():
			p.requires_grad_(False)
		self.preprocess = _make_preprocess(self.image_size)

		self.pos_templates = POS_TEMPLATES
		self.neg_templates = NEG_TEMPLATES
		self._txt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
		self._gen_neg_proto: torch.Tensor | None = None

	# --- tokenize texts ---
	def _tokenize(self, texts: List[str]) -> torch.Tensor:
		ids = [self.tokenizer.encode(t) for t in texts]
		pad = getattr(self.tokenizer, "pad_id", 0)
		max_len = max(len(x) for x in ids)
		padded = [x + [pad] * (max_len - len(x)) for x in ids]
		return torch.tensor(padded, dtype=torch.long, device=self.device)

	# --- encode text features ---
	def _encode_texts(self, texts: List[str]) -> torch.Tensor:
		toks = self._tokenize(texts)
		with torch.inference_mode():
			txt_f = self.model.encode_text(toks, normalize=False)
		return safe_norm(txt_f.float())

	# --- encode image features ---
	def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
		with torch.inference_mode():
			img_f = self.model.encode_image(img_tensor, normalize=False)
		return safe_norm(img_f.float())

	# --- build per-category prototypes ---
	def _get_txt_prototypes(self, category: str) -> Tuple[torch.Tensor, torch.Tensor]:
		key = (category or "unsafe").lower()
		if key in self._txt_cache:
			return self._txt_cache[key]
		pos_texts = [t.format(cat=key) for t in self.pos_templates]
		pos_proto = safe_norm(self._encode_texts(pos_texts).mean(dim=0, keepdim=True))
		self._gen_neg_proto = safe_norm(self._encode_texts(self.neg_templates).mean(dim=0, keepdim=True))
		neg_proto = self._gen_neg_proto
		self._txt_cache[key] = (pos_proto, neg_proto)
		return pos_proto, neg_proto

	# --- public API ---
	def predict_proba(self, pil_img, category):
		pil_img = img_rgba_to_rgb(pil_img)
		img = self.preprocess(pil_img).unsqueeze(0).to(self.device, non_blocking=True)
		img_f = self._encode_image(img)
		pos_proto, neg_proto = self._get_txt_prototypes(category)
		margin = cosine_margin(img_f, pos_proto, neg_proto)
		scale = get_logit_scale(self.model)
		p = torch.sigmoid(scale * margin)
		return float(p.item())

	@torch.inference_mode()
	def predict_proba_batch(self, pil_list, categories):
		xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
		x = torch.stack(xs, dim=0)
		x = x.pin_memory().to(self.device, non_blocking=True, memory_format=torch.channels_last)
		img_f = self._encode_image(x)
		pos_list, neg_list = [], []
		for cat in categories:
			pp, nn = self._get_txt_prototypes(cat)
			pos_list.append(pp)
			neg_list.append(nn)
		pos_stack = torch.cat(pos_list, dim=0)
		neg_stack = torch.cat(neg_list, dim=0)
		pos_sim = (img_f * pos_stack).sum(dim=1)          # (B,)
		neg_sim = (img_f * neg_stack).sum(dim=1)          # (B,)
		margin = pos_sim - neg_sim                        # (B,)
		scale = float(get_logit_scale(self.model) or 1.0)
		p = torch.sigmoid(scale * margin)
		return p.cpu().tolist()
