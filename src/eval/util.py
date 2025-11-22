import torch
from PIL import Image

def dtype_from_str(s: str):
	return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, torch.float16)

def img_rgba_to_rgb(img, bg_color=(0, 0, 0)):
	if img.mode == "P":
		if "transparency" in getattr(img, "info", {}):
			img = img.convert("RGBA")
		else:
			return img.convert("RGB")
	if img.mode in ("LA", "RGBA"):
		if img.mode == "LA":
			img = img.convert("RGBA")
		bg = Image.new("RGB", img.size, bg_color)
		bg.paste(img, mask=img.getchannel("A"))
		return bg
	return img if img.mode == "RGB" else img.convert("RGB")

def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
	return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

def cosine_margin(img_f: torch.Tensor, pos_proto: torch.Tensor, neg_proto: torch.Tensor) -> torch.Tensor:
	assert img_f.dim()==2 and pos_proto.dim()==2 and neg_proto.dim()==2
	assert img_f.size(1) == pos_proto.size(1) == neg_proto.size(1)
	if pos_proto.size(0) == 1 and neg_proto.size(0) == 1:
		s_pos = (img_f * pos_proto).sum(dim=1)
		s_neg = (img_f * neg_proto).sum(dim=1)
	else:
		assert pos_proto.size(0) == img_f.size(0) == neg_proto.size(0), "row-wise prototypes required"
		s_pos = (img_f * pos_proto).sum(dim=1)
		s_neg = (img_f * neg_proto).sum(dim=1)
	return s_pos - s_neg  # (B,)


def get_logit_scale(model, max_val: float = 100.0) -> torch.Tensor:
	s = getattr(model, "logit_scale", None)
	if torch.is_tensor(s):
		try:
			return s.exp().clamp(max=max_val)
		except Exception:
			pass
	return torch.tensor(1.0, device=getattr(model, "device", None))