# src/models/siglip_lp.py
import torch
from typing import List
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoTokenizer

from src.eval.util import img_rgba_to_rgb


def l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization (keeps direction, removes scale)."""
    return x / (x.norm(dim=1, keepdim=True) + eps)

class SigLIPLinearProbe:
    """
    SigLIP image encoder + a learned linear head (w, b).
    Decisions are made in *logit* space; select thresholds on validation logits.

    - L2-normalize image embeddings (optional) — must match training choice.
    - AMP for encode; cast back to fp32 before the linear head.
    - channels_last + non_blocking copies for speed.
    """

    def __init__(self,
                 model_id: str = "google/siglip-so400m-patch14-384",
                 head_path: str | None = None,
                 l2norm: bool = True,
                 use_amp: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # SigLIP backbone (HuggingFace)
        self.amp_dtype = torch.float16 if (self.device == "cuda") else torch.float32
        self.model = SiglipModel.from_pretrained(
            model_id, torch_dtype=self.amp_dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # (opsiyonel) PyTorch 2.x derleme denemesi — başarısız olursa sessizce geç
        self._compiled = False
        if self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                self._compiled = True
            except Exception:
                pass

        #        # >>> Değişiklik: AutoProcessor yerine ayrı bileşenler
        self.image_proc = AutoImageProcessor.from_pretrained(
            model_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )

        # Load linear head weights (npz with 'w' [D,1] and 'b' [1] or a torch .pt/.pth)
        if head_path is None:
            raise ValueError("head_path must be provided (npz with keys 'w' and 'b').")
        if head_path.endswith(".pt") or head_path.endswith(".pth"):
            state = torch.load(head_path, map_location="cpu")
            self.w = state["w"].to(self.device).float()  # [D,1]
            self.b = state["b"].to(self.device).float()  # [1]
        else:
            import numpy as np
            npz = np.load(head_path)
            self.w = torch.from_numpy(npz["w"]).to(self.device).float()  # [D,1]
            self.b = torch.from_numpy(npz["b"]).to(self.device).float()  # [1]

        self.l2norm = bool(l2norm)
        self.use_amp = bool(use_amp)

        # perf flags
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    @torch.inference_mode()
    def _encode(self, pil_list: List) -> torch.Tensor:
        """
        Encode a list of PIL images with SigLIP. Optionally L2-normalize features.
        """
        # Processor builds pixel_values with the exact SigLIP normalization
        imgs = [im.convert("RGB") for im in pil_list]
        enc = self.image_proc(images=[img_rgba_to_rgb(im) for im in pil_list], return_tensors="pt")
        x = enc["pixel_values"]

        # --- HIZ: CPU tarafında pin + ön-dtype, sonra non_blocking + channels_last ---
        if self.device == "cuda":
            # AMP seçilmişse CPU'da da aynı dtype'a çevir (bandwidth daha az)
            if self.use_amp and self.amp_dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.amp_dtype)
            x = x.pin_memory()
            x = x.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            # CPU yolu (AMP yok); contiguous + channels_last (etkisi sınırlı ama zararsız)
            x = x.contiguous().to(memory_format=torch.channels_last)

        if self.use_amp and self.device == "cuda":
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                f = self.model.get_image_features(pixel_values=x)  # [B, D], AMP compute
        else:
            f = self.model.get_image_features(pixel_values=x)

        f = f.float()  # linear head in fp32
        return l2norm_rows(f) if self.l2norm else f

    # -------- Public API --------

    @torch.inference_mode()
    def predict_logit_batch(self, pil_list: List, categories=None) -> list[float]:
        """
        Returns raw logits: z = f @ w + b. Threshold in *logit* space for decisions.
        """
        f = self._encode(pil_list)                   # [B, D]
        logit = (f @ self.w + self.b).squeeze(1)     # [B]
        return logit.detach().cpu().tolist()

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List, categories=None) -> list[float]:
        """
        Returns probabilities via sigmoid(logit). Use for reporting.
        For decisions, prefer a threshold on logits.
        """
        f = self._encode(pil_list)
        logit = (f @ self.w + self.b).squeeze(1)
        logit = torch.clamp(logit, -50, 50)          # numeric stability
        p = torch.sigmoid(logit)
        return p.detach().cpu().tolist()

    @torch.inference_mode()
    def predict_proba(self, pil_img, _category=None) -> float:
        """Single-image probability (sigmoid(logit))."""
        return self.predict_proba_batch([pil_img])[0]
