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
    # ckpt içinde class listesi farklı anahtarlardan gelebilir
    for k in ("class_names", "custom_names", "name", "names", "categories"):
        if isinstance(ckpt.get(k), list) and len(ckpt[k]) > 0:
            return [str(x) for x in ckpt[k]]
    return _FALLBACK_NAMES


def _index_or_minus(names: List[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError:
        return -1


def _load_state_dict_smart(ckpt) -> dict:
    """
    Akıllı state_dict yükleyici - farklı checkpoint formatlarını destekler
    """
    # Öncelik sırası: state_dict → model_state → doğrudan state dict
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    elif "model_state" in ckpt:
        return ckpt["model_state"]
    else:
        # Checkpoint doğrudan state dict olabilir
        # Model parametre anahtarlarını kontrol et
        for key in ckpt.keys():
            if any(param_key in key for param_key in ['backbone', 'head', 'custom_head']):
                return ckpt  # Doğrudan state dict
        raise KeyError("Checkpoint'te 'state_dict' veya 'model_state' anahtarı bulunamadı")


class _AnimeEva02MultiTaskHead(nn.Module):
    """Eğitimdeki modelle TAM UYUMLU versiyon"""

    def __init__(self, repo_id: str, num_out: int):
        super().__init__()
        self.backbone = timm.create_model(f"hf-hub:{repo_id}",
                                          pretrained=False)  # pretrained=False çünkü checkpoint'ten yüklenecek
        cfg = resolve_model_data_config(self.backbone)
        self.preprocess = create_transform(**cfg)

        # Backbone'un head'ini sıfırla - EĞİTİMDEKİ GİBİ
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)

        # Özellik boyutunu keşfet - EĞİTİMDEKİ YÖNTEMLE AYNI
        with torch.no_grad():
            in_size = cfg.get("input_size", (3, 448, 448))
            h, w = int(in_size[-2]), int(in_size[-1])
            fx = self.backbone.forward_features(torch.zeros(1, 3, h, w))
            pre = self.backbone.forward_head(fx, pre_logits=True)
            feat_dim = pre.shape[-1]

        # EĞİTİMDEKİ İSİMLE AYNI: head (custom_head değil)
        self.head = nn.Linear(feat_dim, num_out)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # EĞİTİMDEKİ FORWARD İLE AYNI
        fx = self.backbone.forward_features(x)
        feat = self.backbone.forward_head(fx, pre_logits=True)
        feat = torch.nn.functional.normalize(feat, dim=1)  # Normalizasyon EKLENDİ
        return self.head(feat)


class AnimetimmEva02Trained:
    """
    KULLANIM:
        m = AnimetimmEva02Trained("outputs/model_animetimm_multitask.pt", dtype="float16")
        p  = m.predict_proba(Image.open("imgs/1.jpg"))                  # unsafe olasılığı
        ps = m.predict_proba_batch([Image.open("imgs/1.jpg")])          # [p_unsafe]

    Not: unsafe = max(p_nudity, p_sexy)
    """

    def __init__(self, model_path: str, dtype: str = "float16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = dtype_from_str(dtype)

        ckpt = torch.load(model_path, map_location="cpu")

        # AKILLI STATE_DICT YÜKLEME
        state_dict = _load_state_dict_smart(ckpt)

        # REPO_ID'yi bul (farklı checkpoint'lerde farklı yerlerde olabilir)
        repo_id = ckpt.get("repo_id", "animetimm/eva02_large_patch14_448.dbv4-full")

        # Sınıf isimleri ve indeksler
        self.names = _pick_custom_names(ckpt)
        self.idx_nudity = _index_or_minus(self.names, "nudity")
        self.idx_sexy = _index_or_minus(self.names, "sexy")
        if self.idx_nudity < 0 or self.idx_sexy < 0:
            raise RuntimeError(f"'nudity' veya 'sexy' sınıfı bulunamadı. names={self.names}")

        # Modeli kur - EĞİTİM MODELİYLE AYNI MİMARİ
        self.model = _AnimeEva02MultiTaskHead(repo_id, num_out=len(self.names))

        # STATE DICT UYUMU İÇİN KEY EŞLEME - GÜNCELLENMİŞ
        state_dict = self._adapt_state_dict_keys(state_dict)

        # STRICT MOD'U KAPAT - gereksiz anahtarları yok say
        self.model.load_state_dict(state_dict, strict=False)

        # ÖNEMLİ: seçilen dtype'a al ve cihaza taşı
        self.model.to(self.torch_dtype).to(self.device).eval()

        # Channel-last bellek yerleşimi
        self.model.backbone = self.model.backbone.to(memory_format=torch.channels_last)

        # Preprocess kısayol
        self.preprocess = self.model.preprocess

    def _adapt_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Farklı eğitim script'lerinden gelen state_dict anahtarlarını uyumlu hale getir
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # "backbone.head" ile ilgili anahtarları ATLA - bunlar timm'in orijinal head'i
            if key.startswith('backbone.head.'):
                continue  # Bunları yükleme

            # Diğer anahtarları olduğu gibi al
            new_state_dict[new_key] = value

        print(f"State dict yüklendi. {len(new_state_dict)} anahtar kullanıldı.")
        print(f"İlk 10 anahtar: {list(new_state_dict.keys())[:10]}")

        return new_state_dict

    def _param_dtype(self):
        # Model parametrelerinin gerçek dtype'ı (fp16/bf16/fp32)
        return next(self.model.parameters()).dtype

    def _encode(self, img: Image.Image) -> torch.Tensor:
        img = img_rgba_to_rgb(img)
        x = self.preprocess(img).unsqueeze(0)  # CPU float32
        # CUDA ise non_blocking işe yarasın diye pin_memory kullan
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
        param_dtype = self._param_dtype()
        amp_dtype = torch.bfloat16 if param_dtype == torch.bfloat16 else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = self.model(x)
        probs = torch.sigmoid(logits)[0]
        p_nudity = float(probs[self.idx_nudity])
        p_sexy = float(probs[self.idx_sexy])
        return p_nudity

    @torch.inference_mode()
    def predict_proba_batch(self, pil_list: List[Image.Image], categories: Optional[List[str]] = None) -> List[float]:
        xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
        x = torch.stack(xs, dim=0)  # CPU float32
        if self.device == "cuda":
            x = x.pin_memory()
        x = x.to(self.device,
                 dtype=self._param_dtype(),
                 non_blocking=True,
                 memory_format=torch.channels_last)

        use_amp = (self.device == "cuda")
        param_dtype = self._param_dtype()
        amp_dtype = torch.bfloat16 if param_dtype == torch.bfloat16 else torch.float16
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = self.model(x)
        probs = torch.sigmoid(logits)
        p_n = probs[:, self.idx_nudity]
        # p_s = probs[:, self.idx_sexy]
        unsafe = p_n.float().cpu().tolist()
        return unsafe

####################################
#
# # src/models/animetimm_trained.py
# import torch, torch.nn as nn
# from typing import List, Optional
# from PIL import Image
# import timm
# from timm.data import resolve_model_data_config, create_transform
#
# from src.eval.util import dtype_from_str, img_rgba_to_rgb
#
# _FALLBACK_NAMES = [
#     "alcohol", "drugs", "weapons", "gambling", "nudity", "sexy", "smoking", "violence"
# ]
#
# def _pick_custom_names(ckpt) -> List[str]:
#     for k in ("class_names", "custom_names", "name", "names", "categories"):
#         if isinstance(ckpt.get(k), list) and len(ckpt[k]) > 0:
#             return [str(x) for x in ckpt[k]]
#     return _FALLBACK_NAMES
#
# def _index_or_minus(names: List[str], key: str) -> int:
#     try:
#         return names.index(key)
#     except ValueError:
#         return -1
#
# def _strip_module_prefix(sd: dict) -> dict:
#     # "module." gibi ön ekleri temizle
#     out = {}
#     for k, v in sd.items():
#         if k.startswith("module."):
#             out[k[7:]] = v
#         else:
#             out[k] = v
#     return out
#
# def _load_state_dict_smart(ckpt) -> dict:
#     # Öncelik: state_dict → model_state → doğrudan sd
#     if "state_dict" in ckpt:
#         return _strip_module_prefix(ckpt["state_dict"])
#     if "model_state" in ckpt:
#         return _strip_module_prefix(ckpt["model_state"])
#     # Doğrudan sd olabilir
#     if isinstance(ckpt, dict):
#         return _strip_module_prefix(ckpt)
#     raise KeyError("Checkpoint'te yüklenebilir state_dict bulunamadı.")
#
# class _AnimeEva02MultiTaskHead(nn.Module):
#     """Eğitim mimarisiyle birebir uyumlu: pre_logits → L2 norm → Linear(head)."""
#     def __init__(self, repo_id: str, num_out: int):
#         super().__init__()
#         self.backbone = timm.create_model(f"hf-hub:{repo_id}", pretrained=False)
#         cfg = resolve_model_data_config(self.backbone)
#         self.preprocess = create_transform(**cfg)
#         if hasattr(self.backbone, "reset_classifier"):
#             self.backbone.reset_classifier(0)
#         with torch.no_grad():
#             in_size = cfg.get("input_size", (3, 448, 448))
#             h, w = int(in_size[-2]), int(in_size[-1])
#             fx = self.backbone.forward_features(torch.zeros(1, 3, h, w))
#             pre = self.backbone.forward_head(fx, pre_logits=True)
#             feat_dim = pre.shape[-1]
#         self.head = nn.Linear(feat_dim, num_out)
#         nn.init.zeros_(self.head.bias)
#
#     def forward(self, x):
#         fx = self.backbone.forward_features(x)
#         feat = self.backbone.forward_head(fx, pre_logits=True)
#         feat = torch.nn.functional.normalize(feat, dim=1)
#         return self.head(feat)
#
# class AnimetimmEva02Trained:
#     """
#     KULLANIM:
#         m = AnimetimmEva02Trained("outputs/model_animetimm_multitask.pt", dtype="float16")
#         p  = m.predict_proba(Image.open("imgs/1.jpg"))                 # unsafe = max(nudity,sexy)
#         ps = m.predict_proba_batch([Image.open("imgs/1.jpg")])         # [unsafe]
#
#     Not: unsafe = max(p_nudity, p_sexy)
#     """
#     def __init__(self, model_path: str, dtype: str = "float16"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         # CPU'da fp16/bf16'a zorlamayı kapat → güvenli fp32
#         self.torch_dtype = (dtype_from_str(dtype) if self.device == "cuda" else torch.float32)
#
#         ckpt = torch.load(model_path, map_location="cpu")
#         sd_in = _load_state_dict_smart(ckpt)
#         repo_id = ckpt.get("repo_id", "animetimm/eva02_large_patch14_448.dbv4-full")
#
#         self.names = _pick_custom_names(ckpt)
#         self.idx_nudity = _index_or_minus(self.names, "nudity")
#         self.idx_sexy   = _index_or_minus(self.names, "sexy")
#         if self.idx_nudity < 0 or self.idx_sexy < 0:
#             raise RuntimeError(f"'nudity' veya 'sexy' sınıfı yok. names={self.names}")
#
#         # Model kur
#         self.model = _AnimeEva02MultiTaskHead(repo_id, num_out=len(self.names))
#
#         # ==== ANAHTAR EŞLEME (en kritik düzeltme) ====
#         sd = {}
#         for k, v in sd_in.items():
#             nk = k
#             # Eğitimde 'custom_head.' kullanılmışsa, modele 'head.' olarak eşle
#             if nk.startswith("custom_head."):
#                 nk = "head." + nk[len("custom_head."):]
#             # Timm orijinal classifier anahtarlarını ATLA
#             if nk.startswith("backbone.head."):
#                 continue
#             sd[nk] = v
#
#         missing, unexpected = self.model.load_state_dict(sd, strict=False)
#         # Hızlı sağlık kontrolü: head ağırlığı geldi mi?
#         try:
#             w_norm = float(self.model.head.weight.data.norm().cpu())
#         except Exception:
#             w_norm = -1.0
#         if w_norm < 1e-8:
#             print("[WARN] head.weight norm≈0 görünüyor. Muhtemel eşleme/ckpt uyumsuzluğu → 0.5 çıkışları alırsın.")
#
#         if missing:
#             print(f"[load] missing keys (kısaltılmış): {missing[:10]} (+{max(0,len(missing)-10)} daha)")
#         if unexpected:
#             print(f"[load] unexpected keys (kısaltılmış): {unexpected[:10]} (+{max(0,len(unexpected)-10)} daha)")
#
#         # Cihaza & dtype'a taşı
#         self.model.to(self.torch_dtype).to(self.device).eval()
#         self.model.backbone = self.model.backbone.to(memory_format=torch.channels_last)
#         self.preprocess = self.model.preprocess
#
#     def _param_dtype(self):
#         return next(self.model.parameters()).dtype
#
#     def _encode(self, img: Image.Image) -> torch.Tensor:
#         img = img_rgba_to_rgb(img)
#         x = self.preprocess(img).unsqueeze(0)  # CPU float32
#         if self.device == "cuda":
#             x = x.pin_memory()
#         return x.to(self.device,
#                     dtype=self._param_dtype(),
#                     non_blocking=True,
#                     memory_format=torch.channels_last)
#
#     @torch.inference_mode()
#     def predict_proba(self, pil_img: Image.Image) -> float:
#         x = self._encode(pil_img)
#         use_amp = (self.device == "cuda")
#         # CPU'da autocast kapalı, CUDA'da seçilen dtype
#         amp_dtype = torch.bfloat16 if self._param_dtype() == torch.bfloat16 else torch.float16
#         with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
#             logits = self.model(x)
#         probs = torch.sigmoid(logits)[0]
#         p_nudity = float(probs[self.idx_nudity])
#         p_sexy   = float(probs[self.idx_sexy])
#         # Docstring ile tutarlı: unsafe = max(nudity, sexy)
#         return p_nudity
#
#     @torch.inference_mode()
#     def predict_proba_batch(self, pil_list: List[Image.Image], categories: Optional[List[str]] = None) -> List[float]:
#         xs = [self.preprocess(img_rgba_to_rgb(im)) for im in pil_list]
#         x = torch.stack(xs, dim=0)
#         if self.device == "cuda":
#             x = x.pin_memory()
#         x = x.to(self.device,
#                  dtype=self._param_dtype(),
#                  non_blocking=True,
#                  memory_format=torch.channels_last)
#         use_amp = (self.device == "cuda")
#         amp_dtype = torch.bfloat16 if self._param_dtype() == torch.bfloat16 else torch.float16
#         with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
#             logits = self.model(x)
#         probs = torch.sigmoid(logits)
#         p_n = probs[:, self.idx_nudity]
#         # p_s = probs[:, self.idx_sexy]
#         unsafe = p_n.float().cpu().tolist()
#         return unsafe
