# src/train_multilabel_eva_autostop.py
import os, json, time, random, math, traceback
import numpy as np
import pandas as pd
from PIL import Image
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform

# ======================== KULLANICI AYARLARI ========================
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"
IMG_ROOT = "../data"

CLASS_NAMES = ["alcohol", "drugs", "weapons", "gambling", "nudity", "sexy", "smoking", "violence"]
MASK_NAMES = [f"mask_{c}" for c in CLASS_NAMES]

# EVA tabanlı timm modeli
# REPO_ID = "animetimm/eva02_large_patch14_448.dbv4-full"
REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 41
EPOCHS_HINT = 24  # scheduler için üst sınır; oto-stop zaten kesecek
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4   # UNFREEZE sonrası backbone LR
WEIGHT_DECAY = 5e-2
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
PIN_MEMORY = True

# HIZ & KARARLILIK
USE_AMP_BF16 = True
CLIP_GRAD_NORM = 1.0
NAN_SKIP = True

# DataLoader sağlamlık
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
DL_TIMEOUT_SEC = 600

# Checkpoint & Log
OUT_DIR = "../wdeva_multitask"
os.makedirs(OUT_DIR, exist_ok=True)
LAST_CKPT = os.path.join(OUT_DIR, "last.pt")
LOG_FILE = os.path.join(OUT_DIR, "log.txt")
SAVE_EVERY_EPOCH = True
LOG_EVERY = 25

# ---- OTO DURDURMA / OTO UNFREEZE ----
PLATEAU_DELTA = 0.005  # val macro-F1 artışı bu eşikten küçükse gelişme yok
PATIENCE = 1          # head-only fazında plateau sayısı → UNFREEZE tetikler
MIN_EPOCHS = 3        # çok erken kesmeyi önle
REGRESSION_DELTA = 0.03  # val macro-F1, best'ten bu kadar düşerse "gerileme" say
STOP_ON_REGRESSION = True

UNFREEZE_LAST_BLOCKS = 2  # açılacak ViT blok sayısı
UNFREEZE_MIN_EPOCH = 1    # en az şu epoch görülsün; plateau + bu koşul → unfreeze
UNFREEZE_PATIENCE = 2     # unfreeze fazında plateau sonrası durdurma sabrı
UNFREEZE_PLATEAU_DELTA = 0.003

# ---- TEST TABANLI STOP & OVERFITTING GUARD (YENİ) ----
EVAL_TEST_EVERY        = 1      # her epoch test et; yük ağırsa 2-3 yap
STOP_ON_TEST_REGRESSION= True
TEST_REGRESSION_DELTA  = 0.02   # test macro-F1 best_test'ten bu kadar düşünce dur
OVERFIT_GAP_DELTA      = 0.07   # (val_macro - test_macro) >= bu fark overfit say
OVERFIT_PATIENCE       = 2      # gap üst üste bu kadar epoch sürerse dur

# Best-test artifact yolları
BEST_TEST_PT            = os.path.join(OUT_DIR, "best_test.pt")
BEST_TEST_THRESHOLDS_JS = os.path.join(OUT_DIR, "best_test_thresholds.json")
BEST_TEST_REPORT_JS     = os.path.join(OUT_DIR, "best_test.json")
# ====================================================================

# PyTorch hız bayrakları
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def seed_everything(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sigmoid_np_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)


def safe_div(a, b, eps=1e-9): return a / (b + eps)


def ensure_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def cuda_empty_cache():
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def append_log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if os.path.exists(path): os.remove(path)
    os.replace(tmp, path)


def save_checkpoint(epoch, global_step, model, optimizer, scheduler, note="", control_state=None):
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "sched_t": getattr(scheduler, "t", 0),
        "sched_total": getattr(scheduler, "total", None),
        "sched_warm": getattr(scheduler, "warm", None),
        "class_names": CLASS_NAMES,
        "repo_id": REPO_ID,
        "note": note,
        "control_state": control_state,
    }
    atomic_save(ckpt, LAST_CKPT)
    ep_path = os.path.join(OUT_DIR, f"epoch_{epoch:03d}.pt")
    atomic_save(ckpt, ep_path)
    append_log(f"[ckpt] Kaydedildi: {LAST_CKPT} (epoch={epoch}, step={global_step})")


def try_resume(model, optimizer, scheduler):
    if not os.path.isfile(LAST_CKPT):
        append_log("[resume] Bulunamadı, sıfırdan başlayacak.")
        return 1, 0, None
    append_log(f"[resume] Yükleniyor: {LAST_CKPT}")
    ckpt = torch.load(LAST_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    try:
        optimizer.load_state_dict(ckpt["opt_state"])
    except Exception:
        pass
    if ckpt.get("sched_t") is not None:
        scheduler.t = int(ckpt["sched_t"])
    control_state = ckpt.get("control_state", None)
    append_log(f"[resume] epoch={ckpt['epoch']} global_step={ckpt.get('global_step', 0)} not: {ckpt.get('note', '')}")
    return int(ckpt["epoch"]) + 1, int(ckpt.get("global_step", 0)), control_state


# ----------------- ASL -----------------
try:
    from timm.loss import AsymmetricLossMultiLabel
    def build_loss():
        return AsymmetricLossMultiLabel(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
except Exception:
    class AsymmetricLossMultiLabel(nn.Module):
        def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
            super().__init__(); self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps
        def forward(self, logits, targets):
            x_sig = torch.sigmoid(logits)
            xs_pos = x_sig
            xs_neg = 1.0 - x_sig
            if self.clip is not None and self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1.0)
            los_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gp)
            los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * (xs_pos ** self.gn)
            return -(los_pos + los_neg)
    def build_loss():
        return AsymmetricLossMultiLabel(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)


# ----------------- selected_tags alma -----------------
def maybe_load_selected_tags(repo_id: str):
    names = None
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["selected_tags.csv", "selected_tags.json", "wd_labels.csv", "tags.csv", "tag_names.json"]:
            try:
                p = hf_hub_download(repo_id, fname)
                if fname.endswith(".csv"):
                    df = pd.read_csv(p)
                    if df.shape[1] == 1:
                        names = df.iloc[:, 0].astype(str).tolist()
                    else:
                        for col in ["name", "tag", "label", "labels", "tags"]:
                            if col in df.columns:
                                names = df[col].astype(str).tolist(); break
                elif fname.endswith(".json"):
                    with open(p, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        for k in ["selected_tags", "tags", "names", "labels"]:
                            if k in obj and isinstance(obj[k], list):
                                names = [str(x) for x in obj[k]]; break
                    elif isinstance(obj, list):
                        names = [str(x) for x in obj]
                if names: break
            except Exception:
                continue
    except Exception:
        pass
    return names


# ----------------- Model (ORIJINAL TAGS KORUNACAK) -----------------
class EVAMultiLabelModel(nn.Module):
    def __init__(self, repo_id, num_classes):
        super().__init__()
        self.repo_id = repo_id
        self.backbone = timm.create_model(f"hf-hub:{repo_id}", pretrained=True)
        cfg = resolve_model_data_config(self.backbone)
        self.preprocess = create_transform(**cfg)

        # Orijinal classifier'ı al ve sonra sıfırla (ağırlıkları kopyalamak için yedekle)
        self.original_classifier = None
        orig_cls = None
        if hasattr(self.backbone, "get_classifier"):
            try:
                orig_cls = self.backbone.get_classifier()
            except Exception:
                pass
        if orig_cls is None:
            orig_cls = getattr(self.backbone, "classifier", None)
        if isinstance(orig_cls, nn.Linear):
            self.original_classifier = orig_cls

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)

        with torch.no_grad():
            in_size = cfg.get("input_size", (3, 448, 448))
            h, w = int(in_size[-2]), int(in_size[-1])
            fx = self.backbone.forward_features(torch.zeros(1, 3, h, w))
            pre = self.backbone.forward_head(fx, pre_logits=True)
            feat_dim = pre.shape[-1]

        # Yeni head (custom) + orijinal head
        self.custom_head = nn.Linear(feat_dim, num_classes)
        nn.init.zeros_(self.custom_head.bias)

        self.original_num_classes = self._infer_original_num_classes(self.original_classifier)
        self.original_head = nn.Linear(feat_dim, self.original_num_classes)

        # Ağırlıkları kopyala
        if isinstance(self.original_classifier, nn.Linear):
            if (self.original_classifier.weight.shape == self.original_head.weight.shape and
                self.original_classifier.bias.shape   == self.original_head.bias.shape):
                with torch.no_grad():
                    self.original_head.weight.copy_(self.original_classifier.weight)
                    self.original_head.bias.copy_(self.original_classifier.bias)
            else:
                print("[model] Uyarı: orijinal head shape uyuşmuyor, random init kullanılacak.")

    def _infer_original_num_classes(self, orig_cls):
        if isinstance(orig_cls, nn.Linear):
            return int(orig_cls.out_features)
        n = getattr(self.backbone, "num_classes", None)
        if isinstance(n, int) and n > 0:
            return n
        try:
            tags = maybe_load_selected_tags(self.repo_id)
            if tags and len(tags) > 0:
                return int(len(tags))
        except Exception:
            pass
        print("[model] Uyarı: orijinal sınıf sayısı belirlenemedi, 12000 varsayıldı.")
        return 12000

    def forward(self, x):
        fx = self.backbone.forward_features(x)
        feat_raw = self.backbone.forward_head(fx, pre_logits=True)      # orijinal temsil
        feat_norm = torch.nn.functional.normalize(feat_raw, dim=1)      # custom için normalize
        custom_logits   = self.custom_head(feat_norm)
        original_logits = self.original_head(feat_raw)
        return custom_logits, original_logits


def set_trainable_head_only(model: EVAMultiLabelModel):
    for p in model.backbone.parameters(): p.requires_grad_(False)
    for p in model.custom_head.parameters(): p.requires_grad_(True)
    for p in model.original_head.parameters(): p.requires_grad_(False)

def set_trainable_unfreeze_last_blocks(model: EVAMultiLabelModel, last_blocks: int):
    for p in model.backbone.parameters(): p.requires_grad_(False)
    for p in model.original_head.parameters(): p.requires_grad_(False)
    # son N blok aç
    if hasattr(model.backbone, "blocks") and isinstance(model.backbone.blocks, (list, nn.ModuleList, nn.Sequential)):
        for p in model.backbone.blocks[-last_blocks:].parameters():
            p.requires_grad_(True)
    else:
        children = list(model.backbone.named_children())
        for _, m in children[-last_blocks:]:
            for p in m.parameters():
                p.requires_grad_(True)
    for p in model.custom_head.parameters(): p.requires_grad_(True)


def count_trainable(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_tr, n_all


# ----------------- Scheduler -----------------
class CosineWithWarmup:
    def __init__(self, optimizer, epochs, warmup_epochs, iters_per_epoch, min_lr=1e-6):
        self.opt = optimizer
        self.total = max(1, epochs * max(1, iters_per_epoch))
        self.warm = warmup_epochs * max(1, iters_per_epoch)
        self.t = 0; self.min_lr = min_lr
        for pg in self.opt.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg.get('lr', 1e-3)

    def step(self):
        self.t += 1
        for pg in self.opt.param_groups:
            base = pg['initial_lr']
            if self.warm > 0 and self.t <= self.warm:
                lr = base * self.t / self.warm
            else:
                denom = max(1, (self.total - self.warm))
                tt = min(1.0, (self.t - self.warm) / denom)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * tt))
            pg['lr'] = lr


# ----------------- Val/Test -----------------
def pick_thresholds(val_loader, model):
    append_log("[val] threshold seçimi başlıyor…")
    t0 = time.time()
    model.eval()
    Ls, Ys, Ms = [], [], []
    with torch.inference_mode():
        for bi, (x, y, m, _) in enumerate(val_loader, start=1):
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if (USE_AMP_BF16 and DEVICE == "cuda") else nullcontext())
            with ctx:
                custom_logits, _ = model(x)
                g = custom_logits.float().cpu()
            Ls.append(g); Ys.append(y); Ms.append(m)
            if bi % 50 == 0: append_log(f"[val] {bi} batch işlendi…")
    L = torch.cat(Ls).numpy(); Y = torch.cat(Ys).numpy(); M = torch.cat(Ms).numpy()
    P = sigmoid_np_stable(L)
    C = L.shape[1]
    ths = np.zeros(C, dtype=np.float32)
    grid = np.linspace(0.01, 0.99, 99)
    for ci in range(C):
        mask = M[:, ci] > 0.5
        if mask.sum() == 0: ths[ci] = 0.5; continue
        y = Y[mask, ci].astype(np.uint8); p = P[mask, ci]
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            yhat = (p >= t).astype(np.uint8)
            tp = int((yhat & (y == 1)).sum())
            fp = int((yhat & (y == 0)).sum())
            fn = int(((1 - yhat) & (y == 1)).sum())
            prec = safe_div(tp, tp + fp); rec = safe_div(tp, tp + fn)
            f1 = safe_div(2 * prec * rec, prec + rec)
            if f1 > best_f1: best_f1, best_t = f1, t
        ths[ci] = best_t
    append_log(f"[val] threshold seçimi bitti. süre={time.time() - t0:.1f}s")
    cuda_empty_cache()
    return ths


def evaluate(loader, model, thresholds=None):
    append_log("[eval] değerlendirme…")
    t0 = time.time()
    model.eval()
    Ls, Ys, Ms = [], [], []
    with torch.inference_mode():
        for bi, (x, y, m, _) in enumerate(loader, start=1):
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if (USE_AMP_BF16 and DEVICE == "cuda") else nullcontext())
            with ctx:
                custom_logits, _ = model(x)
                g = custom_logits.float().cpu()
            Ls.append(g); Ys.append(y); Ms.append(m)
            if bi % 50 == 0: append_log(f"[eval] {bi} batch işlendi…")
    L = torch.cat(Ls).numpy(); Y = torch.cat(Ys).numpy(); M = torch.cat(Ms).numpy()
    P = sigmoid_np_stable(L)
    if thresholds is None:
        thresholds = np.array([0.5] * len(CLASS_NAMES), dtype=np.float32)
    YH = (P >= thresholds[None, :]).astype(np.uint8)

    per_class = []
    micro_tp = micro_fp = micro_fn = 0
    for ci, cname in enumerate(CLASS_NAMES):
        mask = M[:, ci] > 0.5
        if mask.sum() == 0:
            per_class.append({"class": cname, "support": 0, "prec": None, "rec": None, "f1": None})
            continue
        y = Y[mask, ci].astype(np.uint8)
        yh = YH[mask, ci].astype(np.uint8)
        tp = int((yh & (y == 1)).sum())
        fp = int((yh & (y == 0)).sum())
        fn = int(((1 - yh) & (y == 1)).sum())
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        prec = safe_div(tp, tp + fp); rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec)
        per_class.append({"class": cname, "support": int(mask.sum()),
                          "prec": float(prec), "rec": float(rec), "f1": float(f1)})
    f1s = [d["f1"] for d in per_class if d["f1"] is not None]
    macro_f1 = float(np.mean(f1s)) if f1s else None
    micro_prec = safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)
    append_log(f"[eval] süre={time.time() - t0:.1f}s microF1={micro_f1:.4f} macroF1={macro_f1:.4f}")
    cuda_empty_cache()
    return per_class, {"macro_f1": macro_f1,
                       "micro_prec": float(micro_prec),
                       "micro_rec": float(micro_rec),
                       "micro_f1": float(micro_f1)}


# ----------------- Dataset -----------------
class MultiLabelCsvDataset(Dataset):
    def __init__(self, csv_path, root_dir, preprocess):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.pp = preprocess

        expect = ["name"] + CLASS_NAMES + MASK_NAMES
        for c in expect:
            if c not in self.df.columns:
                raise ValueError(f"CSV kolon eksik: {c}")

        self.names = self.df["name"].astype(str).values
        self.Y = self.df[CLASS_NAMES].astype("float32").values
        self.M = self.df[MASK_NAMES].astype("float32").values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.names[idx])
        img = Image.open(path)
        img = ensure_rgb(img)
        x = self.pp(img)
        y = torch.from_numpy(self.Y[idx])
        m = torch.from_numpy(self.M[idx])
        return x, y, m, path


# ----------------- Optimizer yardımcıları -----------------
def build_optimizer_head_only(model: EVAMultiLabelModel):
    head_params = [p for p in model.custom_head.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [{"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY
    )


def build_optimizer_unfreeze(model: EVAMultiLabelModel):
    head_params = [p for p in model.custom_head.parameters() if p.requires_grad]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR_HEAD, "initial_lr": LR_HEAD},
            {"params": bb_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY
    )


def build_scheduler(optimizer, iters_per_epoch, total_epochs=EPOCHS_HINT, warmup=WARMUP_EPOCHS):
    return CosineWithWarmup(optimizer, epochs=total_epochs,
                            warmup_epochs=warmup, iters_per_epoch=iters_per_epoch, min_lr=1e-6)


# ----------------- Inference için yardımcı fonksiyon -----------------
def predict_with_both_heads(model, image_tensor):
    model.eval()
    with torch.no_grad():
        custom_logits, original_logits = model(image_tensor)
        custom_probs = torch.sigmoid(custom_logits)
        original_probs = torch.sigmoid(original_logits)
    return custom_probs, original_probs


# ----------------- Eğitim -----------------
def train():
    seed_everything(SEED)
    append_log(">> EVA backbone yükleniyor...")
    tmp = timm.create_model(f"hf-hub:{REPO_ID}", pretrained=True)
    cfg = resolve_model_data_config(tmp)
    preprocess = create_transform(**cfg); del tmp

    # DataLoaders
    dl_common = dict(
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        timeout=DL_TIMEOUT_SEC
    )
    if NUM_WORKERS > 0: dl_common["prefetch_factor"] = PREFETCH_FACTOR

    train_ds = MultiLabelCsvDataset(TRAIN_CSV, IMG_ROOT, preprocess)
    val_ds   = MultiLabelCsvDataset(VAL_CSV,   IMG_ROOT, preprocess)
    test_ds  = MultiLabelCsvDataset(TEST_CSV,  IMG_ROOT, preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **dl_common)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_common)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_common)
    append_log(f"[dl] train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

    # Model
    model = EVAMultiLabelModel(REPO_ID, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.backbone = model.backbone.to(DEVICE).to(memory_format=torch.channels_last)
    model.custom_head = model.custom_head.to(DEVICE)
    model.original_head = model.original_head.to(DEVICE)
    if hasattr(model.backbone, "set_grad_checkpointing"):
        try: model.backbone.set_grad_checkpointing(True)
        except Exception: pass

    # selected_tags
    sel_tags = maybe_load_selected_tags(REPO_ID)
    if sel_tags:
        with open(os.path.join(OUT_DIR, "selected_tags.json"), "w", encoding="utf-8") as f:
            json.dump(sel_tags, f, ensure_ascii=False, indent=2)
        append_log(f"[tags] {len(sel_tags)} orijinal tag yüklendi")
    else:
        append_log("[tags] Orijinal tagler yüklenemedi")

    # Faz kontrol durumu
    control = {
        "phase": "head",  # "head" → (gerekirse) "unfreeze"
        "best_macro_f1": -1.0,
        "best_micro_f1": -1.0,
        "best_val_thresholds": None,
        "last_val_macro_f1": None,
        "plateau": 0,
        "total_epochs": 0,
        "unfreeze_epoch": None,
        # --- yeni: test & overfit takibi ---
        "best_test_macro_f1": -1.0,
        "last_test_macro_f1": None,
        "overfit_streak": 0
    }

    # Başlangıç: HEAD-ONLY
    set_trainable_head_only(model)
    n_tr, n_all = count_trainable(model)
    append_log(f"[model] trainable params={n_tr}/{n_all} (head-only)")

    optimizer = build_optimizer_head_only(model)
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, iters_per_epoch)

    start_epoch, global_step, resumed_control = try_resume(model, optimizer, scheduler)
    if resumed_control is not None:
        # yeni alanlar yoksa korunur; varsa güncellenir
        control.update({k: resumed_control.get(k, control[k]) for k in control.keys()})
        append_log(f"[control] resume state: {control}")

        if control["phase"] == "unfreeze":
            set_trainable_unfreeze_last_blocks(model, UNFREEZE_LAST_BLOCKS)
            optimizer = build_optimizer_unfreeze(model)
            scheduler = build_scheduler(optimizer, iters_per_epoch)
            append_log("[resume] optimizer/scheduler UNFREEZE fazı için yeniden kuruldu.")
        else:
            set_trainable_head_only(model)
            optimizer = build_optimizer_head_only(model)
            scheduler = build_scheduler(optimizer, iters_per_epoch)
            append_log("[resume] optimizer/scheduler HEAD fazı için yeniden kuruldu.")

    loss_fn = build_loss()

    append_log(">> Eğitim başlıyor...")
    epoch = start_epoch
    try:
        while True:
            control["total_epochs"] += 1
            model.train()
            run_loss, denom_count = 0.0, 0.0
            t0 = time.time()
            append_log(f"[epoch {epoch:02d}] train loop başlıyor… (phase={control['phase']})")
            cuda_empty_cache()

            for it, (x, y, m, _) in enumerate(train_loader, start=1):
                x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(DEVICE); m = m.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if (USE_AMP_BF16 and DEVICE == "cuda") else nullcontext())
                try:
                    with ctx:
                        custom_logits, _ = model(x)  # sadece custom logits
                        loss_mat = loss_fn(custom_logits, y)
                        denom = m.sum()
                        if denom.item() < 1: continue
                        loss = (loss_mat * m).sum() / denom.clamp_min(1.0)

                    if NAN_SKIP and (not torch.isfinite(loss)):
                        append_log(f"[epoch {epoch:02d}] it={it}/{len(train_loader)} SKIP (non-finite loss)")
                        continue

                    loss.backward()
                    if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()

                except torch.cuda.OutOfMemoryError:
                    append_log(f"[oom] it={it} → batch SKIP; cache temizleniyor")
                    cuda_empty_cache()
                    continue

                if DEVICE == "cuda" and (it % 8 == 0):
                    cuda_empty_cache()

                run_loss += float(loss.item()) * float(denom.item())
                denom_count += float(denom.item())
                global_step += 1

                if it % LOG_EVERY == 0:
                    lrs = [round(pg['lr'], 6) for pg in optimizer.param_groups]
                    append_log(f"[epoch {epoch:02d}] it={it}/{len(train_loader)} "
                               f"loss={loss.item():.4f} step={global_step} lr={lrs}")

            epoch_loss = run_loss / max(denom_count, 1.0)
            dt = time.time() - t0
            lrs = [round(pg['lr'], 6) for pg in optimizer.param_groups]
            append_log(f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | time={dt:.1f}s | lr={lrs}")

            # ---- Val: thresholds -> metrics
            val_thresholds = pick_thresholds(val_loader, model)
            val_per, val_agg = evaluate(val_loader, model, val_thresholds)
            macro = val_agg["macro_f1"] if val_agg["macro_f1"] is not None else -1.0
            micro = val_agg["micro_f1"]

            # val raporu
            ep_rep = {
                "epoch": int(epoch),
                "phase": control["phase"],
                "thresholds": {c: float(t) for c, t in zip(CLASS_NAMES, val_thresholds)},
                "aggregate": val_agg,
                "per_class": val_per
            }
            with open(os.path.join(OUT_DIR, f"val_epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(ep_rep, f, indent=2)

            # best-val güncelle + (eski davranış) best olunca test
            improved = False
            if macro > control["best_macro_f1"] + 1e-12:
                control["best_macro_f1"] = macro
                control["best_micro_f1"] = max(control["best_micro_f1"], micro)
                control["best_val_thresholds"] = {c: float(t) for c, t in zip(CLASS_NAMES, val_thresholds)}
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "repo_id": REPO_ID,
                    "val_macro_f1": float(control["best_macro_f1"]),
                    "val_thresholds": control["best_val_thresholds"]
                }, os.path.join(OUT_DIR, "best_val.pt"))
                with open(os.path.join(OUT_DIR, "best_val_thresholds.json"), "w", encoding="utf-8") as f:
                    json.dump(control["best_val_thresholds"], f, indent=2)
                append_log(f"[val] ✅ best updated | macro-F1={control['best_macro_f1']:.4f}")
                improved = True

                # Hız için testi sadece best-val güncellenince koştur
                best_th_np = np.array([control["best_val_thresholds"][c] for c in CLASS_NAMES], dtype=np.float32)
                test_per, test_agg = evaluate(test_loader, model, best_th_np)
                with open(os.path.join(OUT_DIR, "best_test_on_bestval.json"), "w", encoding="utf-8") as f:
                    json.dump({"per_class": test_per, "aggregate": test_agg}, f, indent=2)
                append_log(f"[test] (on-bestval) macroF1={test_agg['macro_f1']:.4f} microF1={test_agg['micro_f1']:.4f}")

            # plateau / regression güncelle
            last_macro = control["last_val_macro_f1"]
            delta = (macro - last_macro) if (last_macro is not None) else float("inf")
            if improved:
                control["plateau"] = 0
            else:
                if last_macro is not None:
                    delta_thr = PLATEAU_DELTA if control["phase"] == "head" else UNFREEZE_PLATEAU_DELTA
                    if delta < delta_thr:
                        control["plateau"] += 1
            control["last_val_macro_f1"] = macro

            # --- (YENİ) Her epoch test ölç (val_thresholds ile) + best-test kaydet + overfit takibi ---
            test_macro_for_stop = None
            if (EVAL_TEST_EVERY == 1) or (epoch % EVAL_TEST_EVERY == 0):
                test_per_cur, test_agg_cur = evaluate(
                    test_loader, model, np.array(val_thresholds, dtype=np.float32)
                )
                test_macro_for_stop = test_agg_cur["macro_f1"] if test_agg_cur["macro_f1"] is not None else -1.0
                control["last_test_macro_f1"] = test_macro_for_stop

                # diagnostik kayıt
                with open(os.path.join(OUT_DIR, f"val_test_epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "epoch": int(epoch),
                        "val_macro_f1": float(macro),
                        "test_macro_f1": float(test_macro_for_stop),
                        "gap": (float(macro) - float(test_macro_for_stop)) if (test_macro_for_stop is not None and macro is not None) else None
                    }, f, indent=2)

                # best-test güncelle + artifact kaydet
                if test_macro_for_stop > control["best_test_macro_f1"] + 1e-12:
                    control["best_test_macro_f1"] = test_macro_for_stop
                    th_map = {c: float(t) for c, t in zip(CLASS_NAMES, val_thresholds)}
                    atomic_save({
                        "state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "repo_id": REPO_ID,
                        "val_thresholds": th_map,
                        "test_macro_f1": float(test_macro_for_stop)
                    }, BEST_TEST_PT)
                    with open(BEST_TEST_THRESHOLDS_JS, "w", encoding="utf-8") as f:
                        json.dump(th_map, f, indent=2)
                    with open(BEST_TEST_REPORT_JS, "w", encoding="utf-8") as f:
                        json.dump({"per_class": test_per_cur, "aggregate": test_agg_cur}, f, indent=2)
                    append_log(f"[test] ✅ best-test updated | macroF1={test_macro_for_stop:.4f} → {BEST_TEST_PT}")

                # overfit gap takibi
                if (macro is not None) and (test_macro_for_stop is not None):
                    gap = macro - test_macro_for_stop
                    if gap >= OVERFIT_GAP_DELTA:
                        control["overfit_streak"] += 1
                        append_log(f"[overfit] gap={gap:.4f} (streak={control['overfit_streak']}/{OVERFIT_PATIENCE})")
                    else:
                        control["overfit_streak"] = 0

            # epoch checkpoint
            if SAVE_EVERY_EPOCH:
                save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                                note="epoch_end", control_state=control)

            # ---- FAZ GEÇİŞİ: HEAD → UNFREEZE (plateau tetikleyici) ----
            if (control["phase"] == "head"
                and control["total_epochs"] >= max(MIN_EPOCHS, UNFREEZE_MIN_EPOCH)
                and control["plateau"] >= PATIENCE):
                append_log(f"[auto] UNFREEZE tetikleniyor: plateau={control['plateau']} (phase=head)")
                set_trainable_unfreeze_last_blocks(model, UNFREEZE_LAST_BLOCKS)
                n_tr, n_all = count_trainable(model)
                append_log(f"[model] trainable params={n_tr}/{n_all} (unfreeze last {UNFREEZE_LAST_BLOCKS})")

                optimizer = build_optimizer_unfreeze(model)
                scheduler = build_scheduler(optimizer, iters_per_epoch)
                control["phase"] = "unfreeze"
                control["plateau"] = 0
                control["unfreeze_epoch"] = int(epoch)
                append_log("[auto] UNFREEZE aktif. Backbone düşük LR ile birlikte eğitime devam.")

            # ---- OTO-DURDURMA (fazlara göre) + (YENİ) test/overfit koşulları ----
            if control["phase"] == "head":
                stop_regress = STOP_ON_REGRESSION and (macro < control["best_macro_f1"] - REGRESSION_DELTA) and (control["total_epochs"] >= MIN_EPOCHS)
                stop_plateau = False
            else:
                stop_regress = STOP_ON_REGRESSION and (macro < control["best_macro_f1"] - REGRESSION_DELTA) and (control["total_epochs"] >= MIN_EPOCHS)
                stop_plateau = (control["plateau"] >= UNFREEZE_PATIENCE) and (control["total_epochs"] >= MIN_EPOCHS)

            stop_test_regress = False
            if STOP_ON_TEST_REGRESSION and (test_macro_for_stop is not None) and (control["best_test_macro_f1"] > 0):
                stop_test_regress = (test_macro_for_stop < control["best_test_macro_f1"] - TEST_REGRESSION_DELTA) and (control["total_epochs"] >= MIN_EPOCHS)

            stop_overfit = (control["overfit_streak"] >= OVERFIT_PATIENCE)

            if stop_regress:
                append_log(f"[auto] stop (val regression): val_macro={macro:.4f} < best-{REGRESSION_DELTA}={control['best_macro_f1']-REGRESSION_DELTA:.4f}")
                break
            if stop_plateau:
                append_log(f"[auto] stop (val plateau): {control['plateau']} >= {(UNFREEZE_PATIENCE if control['phase']=='unfreeze' else PATIENCE)}")
                break
            if stop_test_regress:
                append_log(f"[auto] stop (test regression): test_macro={test_macro_for_stop:.4f} < best-{TEST_REGRESSION_DELTA}={control['best_test_macro_f1']-TEST_REGRESSION_DELTA:.4f}")
                break
            if stop_overfit:
                append_log(f"[auto] stop (overfit gap): streak={control['overfit_streak']} gap≥{OVERFIT_GAP_DELTA}")
                break

            epoch += 1

    except KeyboardInterrupt:
        append_log("\n[train] CTRL+C → checkpoint kaydediliyor…")
        save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                        note="keyboard_interrupt", control_state=control)
        append_log("[train] Güvenle çıkıldı.")
        return
    except Exception as e:
        append_log("[train] HATA! checkpoint kaydediliyor…")
        traceback.print_exc()
        save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                        note=f"exception:{type(e).__name__}", control_state=control)
        raise

    # ---- EN SON: best val threshold ile nihai TEST ----
    append_log(">> Test değerlendirme (final)…")
    best_th_path = os.path.join(OUT_DIR, "best_val_thresholds.json")
    if os.path.isfile(best_th_path) and control["best_val_thresholds"] is not None:
        best_th = np.array([float(control["best_val_thresholds"].get(c, 0.5)) for c in CLASS_NAMES], dtype=np.float32)
    else:
        best_th = None

    per_class, agg = evaluate(test_loader, model, best_th)

    final_path = os.path.join(OUT_DIR, "model_eva_multilabel_autostop.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "repo_id": REPO_ID,
        "config": {
            "batch_size": BATCH_SIZE, "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE, "wd": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "patience": PATIENCE, "plateau_delta": PLATEAU_DELTA,
            "unfreeze_blocks": UNFREEZE_LAST_BLOCKS
        }
    }, final_path)
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({c: float(t) for c, t in zip(CLASS_NAMES, (best_th if best_th is not None else [0.5] * len(CLASS_NAMES)))}, f, indent=2)
    with open(os.path.join(OUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    rep = {"per_class": per_class, "aggregate": agg}
    with open(os.path.join(OUT_DIR, "report_test.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    append_log("\n== TEST Sonuçları ==")
    for d in per_class:
        sup = d["support"]
        if sup == 0:
            append_log(f"{d['class']:<10s} | sup=0   | P=--- | R=--- | F1=---")
        else:
            append_log(f"{d['class']:<10s} | sup={sup:4d} | P={d['prec']:.3f} | R={d['rec']:.3f} | F1={d['f1']:.3f}")
    append_log(f"Macro-F1: {agg['macro_f1']:.3f} | Micro-F1: {agg['micro_f1']:.3f}")
    append_log(f"Kaydedildi: {final_path}, thresholds.json, class_names.json, report_test.json")

    # Son kontrol: best-test artifact'leri yoksa (eğitim çok erken kesildiyse) mevcut modeli yaz
    if not os.path.isfile(BEST_TEST_PT):
        append_log("[final] Uyarı: best_test.pt bulunamadı, mevcut son modeli best-test olarak yazıyorum.")
        th_map = control["best_val_thresholds"] or {c: 0.5 for c in CLASS_NAMES}
        atomic_save({
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "repo_id": REPO_ID,
            "val_thresholds": th_map,
            "note": "fallback_best_test_at_end"
        }, BEST_TEST_PT)
        with open(BEST_TEST_THRESHOLDS_JS, "w", encoding="utf-8") as f:
            json.dump(th_map, f, indent=2)
        with open(BEST_TEST_REPORT_JS, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)

    save_checkpoint(epoch, global_step, model, optimizer, scheduler, note="final", control_state=control)


if __name__ == "__main__":
    train()
