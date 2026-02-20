import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

# Add third_party/hysac to path
_ROOT = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "hysac"
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

try:
    from hysac.models import HySAC
    from hysac import lorentz as L
except ImportError:
    # If not found, user might have removed it. Raises typical error.
    pass

from cmt.utils.torch_utils import get_device, dtype_from_str
from cmt.utils.image_utils import ensure_rgb
from cmt.models.base import Model

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

def _hyp_sims(img_emb, txt_emb, curv):
    if img_emb.dim() == 1: img_emb = img_emb.unsqueeze(0)
    if txt_emb.dim() == 1: txt_emb = txt_emb.unsqueeze(0)
    d = L.pairwise_dist(img_emb, txt_emb, curv=curv)
    return -d

class HySACWrapper(Model):
    def __init__(self, model_id="aimagelab/HySAC", dtype="float16"):
        self.device = get_device()
        self.dtype = dtype_from_str(dtype)
        print(f"Loading HySAC {model_id}...")
        self.model = HySAC.from_pretrained(model_id, device=self.device).to(self.device).eval()
        self.image_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self._txt_cache = {}

    @torch.inference_mode()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = [ensure_rgb(im) for im in images]
        px = self.image_proc(images=imgs, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, self.dtype)
        return self.model.encode_image(px, True)

    @torch.inference_mode()
    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        # Query-dependent logits not fitting standard 'one logit vector per image' 
        # unless 'unsafe' is implied as the default category.
        # This wrapper assumes binary "sexual/unsafe" detection if accessed via generic methods
        return self._predict_custom(images, ["sexual"])

    @torch.inference_mode()
    def prob(self, images: List[Image.Image]) -> np.ndarray:
        # Same assumption
        l = self.logits(images)
        return torch.sigmoid(l).cpu().float().numpy()

    # --- HySAC specific method for custom categories ---
    @torch.inference_mode()
    def predict_proba_custom(self, images: List[Image.Image], categories: List[str]) -> np.ndarray:
        # This matches the logic from original hysac_unsafe.py
        img_emb = self.encode(images)
        curv = self.model.curv.exp()
        scale = self.model.logit_scale.exp().clamp(max=100.0)
        
        B = len(images)
        C = len(categories)
        out = torch.zeros((B, C), device=self.device, dtype=torch.float32)

        for i, c in enumerate(categories):
            c = (c or "unsafe").lower()
            if c in self._txt_cache:
                pos_emb, neg_emb = self._txt_cache[c]
            else:
                pos_txt = [t.format(cat=c) for t in POS_TEMPLATES]
                neg_txt = [t.format(cat=c) for t in NEG_TEMPLATES]
                pos_emb = self._encode_text(pos_txt)
                neg_emb = self._encode_text(neg_txt)
                self._txt_cache[c] = (pos_emb, neg_emb)
            
            # Distance -> Logit
            pos_sims = _hyp_sims(img_emb, pos_emb, curv)
            neg_sims = _hyp_sims(img_emb, neg_emb, curv)
            
            # LogSumExp scoring
            pos_logit = torch.logsumexp(scale * pos_sims, dim=1)
            neg_logit = torch.logsumexp(scale * neg_sims, dim=1)
            
            out[:, i] = pos_logit - neg_logit # This is the logit difference
            
        return torch.sigmoid(out).cpu().numpy()
        
    def _encode_text(self, texts: list[str]):
        enc = self.text_tok(texts, padding=True, truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        # HySAC expects list of tensors for encode_text
        tokens = [row.unsqueeze(0) for row in ids]
        
        return self.model.encode_text(list(ids), True)
    
    def _predict_custom(self, images, categories):
        # Helper for logits()
        img_emb = self.encode(images)
        curv = self.model.curv.exp()
        scale = self.model.logit_scale.exp().clamp(max=100.0)
        out = []
        c = "sexual"
        if c in self._txt_cache:
            pos_emb, neg_emb = self._txt_cache[c]
        else:
            pos_txt = [t.format(cat=c) for t in POS_TEMPLATES]
            neg_txt = [t.format(cat=c) for t in NEG_TEMPLATES]
            pos_emb = self._encode_text(pos_txt)
            neg_emb = self._encode_text(neg_txt)
            self._txt_cache[c] = (pos_emb, neg_emb)
        
        pos_sims = _hyp_sims(img_emb, pos_emb, curv)
        neg_sims = _hyp_sims(img_emb, neg_emb, curv)
        pos_logit = torch.logsumexp(scale * pos_sims, dim=1)
        neg_logit = torch.logsumexp(scale * neg_sims, dim=1)
        
        return (pos_logit - neg_logit).unsqueeze(1) # [B, 1]

