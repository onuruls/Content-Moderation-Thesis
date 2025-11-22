# src/models/animetimm_trained.py
import torch, torch.nn as nn
from typing import List, Optional
from PIL import Image
import timm
from timm.data import resolve_model_data_config, create_transform

from src.eval.util import dtype_from_str, img_rgba_to_rgb

_FALLBACK_NAMES = [
    "alcohol", "drugs", "weapons", "gambling", "nudity", "sexy", "smoking", "violence"
]

def _pick_custom_names(ckpt) -> List[str]:
    for k in ("class_names", "custom_names", "name", "names", "categories"):
        if isinstance(ckpt.get(k), list) and len(ckpt[k]) > 0:
            return [str(x) for x in ckpt[k]]
    return _FALLBACK_NAMES

def _index_or_minus(names: List[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError:
        return -1

def _strip_module_prefix(sd: dict) -> dict:
    # "module." gibi ön ekleri temizle
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out

def _load_state_dict_smart(ckpt) -> dict:
    # Öncelik: state_dict → model_state → doğrudan sd
    if "state_dict" in ckpt:
        return _strip_module_prefix(ckpt["state_dict"])
    if "model_state" in ckpt:
        return _strip_module_prefix(ckpt["model_state"])
    # Doğrudan sd olabilir
    if isinstance(ckpt, dict):
        return _strip_module_prefix(ckpt)
    raise KeyError("Checkpoint'te yüklenebilir state_dict bulunamadı.")

class _AnimeEva02MultiTaskHead(nn.Module):
    """Eğitim mimarisiyle birebir uyumlu: pre_logits → L2 norm → Linear(head)."""
    def __init__(self, repo_id: str, num_out: int):
        super().__init__()
        self.backbone = timm.create_model(f"hf-hub:{repo_id}", pretrained=False)
        cfg = resolve_model_data_config(self.backbone)
        self.preprocess = create_transform(**cfg)
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        with torch.no_grad():
            in_size = cfg.get("input_size", (3, 448, 448))
            h, w = int(in_size[-2]), int(in_size[-1])
            fx = self.backbone.forward_features(torch.zeros(1, 3, h, w))
            pre = self.backbone.forward_head(fx, pre_logits=True)
            feat_dim = pre.shape[-1]
        self.head = nn.Linear(feat_dim, num_out)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        fx = self.backbone.forward_features(x)
        feat = self.backbone.forward_head(fx, pre_logits=True)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return self.head(feat)

class WDEva02TaggerTrained:
    """
    KULLANIM:
        m = AnimetimmEva02Trained("outputs/model_animetimm_multitask.pt", dtype="float16")
        p  = m.predict_proba(Image.open("imgs/1.jpg"))                 # unsafe = max(nudity,sexy)
        ps = m.predict_proba_batch([Image.open("imgs/1.jpg")])         # [unsafe]

    Not: unsafe = max(p_nudity, p_sexy)
    """
    def __init__(self, model_path: str, dtype: str = "float16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU'da fp16/bf16'a zorlamayı kapat → güvenli fp32
        self.torch_dtype = (dtype_from_str(dtype) if self.device == "cuda" else torch.float32)

        ckpt = torch.load(model_path, map_location="cpu")
        sd_in = _load_state_dict_smart(ckpt)
        repo_id = ckpt.get("repo_id", "SmilingWolf/wd-eva02-large-tagger-v3")

        self.names = _pick_custom_names(ckpt)
        self.idx_nudity = _index_or_minus(self.names, "nudity")
        self.idx_sexy   = _index_or_minus(self.names, "sexy")
        if self.idx_nudity < 0 or self.idx_sexy < 0:
            raise RuntimeError(f"'nudity' veya 'sexy' sınıfı yok. names={self.names}")

        # Model kur
        self.model = _AnimeEva02MultiTaskHead(repo_id, num_out=len(self.names))

        # ==== ANAHTAR EŞLEME (en kritik düzeltme) ====
        sd = {}
        for k, v in sd_in.items():
            nk = k
            # Eğitimde 'custom_head.' kullanılmışsa, modele 'head.' olarak eşle
            if nk.startswith("custom_head."):
                nk = "head." + nk[len("custom_head."):]
            # Timm orijinal classifier anahtarlarını ATLA
            if nk.startswith("backbone.head."):
                continue
            sd[nk] = v

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # Hızlı sağlık kontrolü: head ağırlığı geldi mi?
        try:
            w_norm = float(self.model.head.weight.data.norm().cpu())
        except Exception:
            w_norm = -1.0
        if w_norm < 1e-8:
            print("[WARN] head.weight norm≈0 görünüyor. Muhtemel eşleme/ckpt uyumsuzluğu → 0.5 çıkışları alırsın.")

        if missing:
            print(f"[load] missing keys (kısaltılmış): {missing[:10]} (+{max(0,len(missing)-10)} daha)")
        if unexpected:
            print(f"[load] unexpected keys (kısaltılmış): {unexpected[:10]} (+{max(0,len(unexpected)-10)} daha)")

        # Cihaza & dtype'a taşı
        self.model.to(self.torch_dtype).to(self.device).eval()
        self.model.backbone = self.model.backbone.to(memory_format=torch.channels_last)
        self.preprocess = self.model.preprocess

    def _param_dtype(self):
        return next(self.model.parameters()).dtype

    def _encode(self, img: Image.Image) -> torch.Tensor:
        img = img_rgba_to_rgb(img)
        x = self.preprocess(img).unsqueeze(0)  # CPU float32
        if self.device == "cuda":
            x = x.pin_memory()
        return x.to(self.device,
                    dtype=self._param_dtype(),
                    non_blocking=True,
                    memory_format=torch.channels_last)

    @torch.inference_mode()
    def predict_proba(self, pil_img: Image.Image) -> float:
        x = self._encode(pil_img)
        use_amp = (self.device == "cuda")
        # CPU'da autocast kapalı, CUDA'da seçilen dtype
        amp_dtype = torch.bfloat16 if self._param_dtype() == torch.bfloat16 else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = self.model(x)
        probs = torch.sigmoid(logits)[0]
        p_nudity = float(probs[self.idx_nudity])
        p_sexy   = float(probs[self.idx_sexy])
        # Docstring ile tutarlı: unsafe = max(nudity, sexy)
        return p_nudity

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List[Image.Image], categories: Optional[List[str]] = None) -> List[float]:
        xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
        x = torch.stack(xs, dim=0)
        if self.device == "cuda":
            x = x.pin_memory()
        x = x.to(self.device,
                 dtype=self._param_dtype(),
                 non_blocking=True,
                 memory_format=torch.channels_last)
        use_amp = (self.device == "cuda")
        amp_dtype = torch.bfloat16 if self._param_dtype() == torch.bfloat16 else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = self.model(x)
        probs = torch.sigmoid(logits)
        p_n = probs[:, self.idx_nudity]
        # p_s = probs[:, self.idx_sexy]
        unsafe = p_n.float().cpu().tolist()
        return unsafe
