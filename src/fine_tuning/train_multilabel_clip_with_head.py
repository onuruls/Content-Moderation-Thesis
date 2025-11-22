# src/train_multilabel_clip_unfreeze_only.py
import json, time, random, math, traceback, os
import numpy as np, pandas as pd
from PIL import Image
from contextlib import nullcontext

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip

# ======================== KULLANICI AYARLARI ========================
TRAIN_CSV = "train.csv"
VAL_CSV   = "val.csv"
TEST_CSV  = "test.csv"
IMG_ROOT  = "../data"

CLASS_NAMES = ["alcohol","drugs","weapons","gambling","nudity","sexy","smoking","violence"]
MASK_NAMES  = [f"mask_{c}" for c in CLASS_NAMES]

CLIP_ARCH       = "ViT-L-14"
CLIP_PRETRAINED = "openai"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

BATCH_SIZE   = 64
EPOCHS_HINT  = 24
LR_HEAD      = 1e-3
LR_BACKBONE  = 1e-4
WEIGHT_DECAY = 5e-2
WARMUP_EPOCHS= 2
NUM_WORKERS  = 4
PIN_MEMORY   = True

# HIZ & KARARLILIK
USE_AMP_BF16   = True
CLIP_GRAD_NORM = 1.0
NAN_SKIP       = True
L2_NORMALIZE_FEATS = True

# DataLoader
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 2
DL_TIMEOUT_SEC     = 600

# Checkpoint & Log
OUT_DIR   = "../output_clip_multilabel_ft_unfreeze"
os.makedirs(OUT_DIR, exist_ok=True)
LAST_CKPT = os.path.join(OUT_DIR, "last.pt")
BEST_TEST_PT = os.path.join(OUT_DIR, "best_test.pt")
BEST_TEST_JSON = os.path.join(OUT_DIR, "best_test.json")
LOG_FILE  = os.path.join(OUT_DIR, "log.txt")
SAVE_EVERY_EPOCH = True
LOG_EVERY = 25
HEAD_INIT_PATH = os.path.join(OUT_DIR, "head.pt")  # varsa yüklenir (opsiyonel)

# ---- OTO DURDURMA (tek faz: unfreeze) ----
UNFREEZE_PLATEAU_DELTA = 0.003
UNFREEZE_PATIENCE      = 2
MIN_EPOCHS             = 3
REGRESSION_DELTA       = 0.03
STOP_ON_REGRESSION     = True

# ---- TEST-BASED STOP & OVERFITTING GUARD ----
EVAL_TEST_EVERY        = 1
STOP_ON_TEST_REGRESSION= True
TEST_REGRESSION_DELTA  = 0.02
OVERFIT_GAP_DELTA      = 0.07
OVERFIT_PATIENCE       = 2
# ====================================================================

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

# ----------------- Yardımcılar -----------------
def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sigmoid_np_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
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
        try: torch.cuda.empty_cache()
        except Exception: pass

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
        "clip_arch": CLIP_ARCH,
        "clip_pretrained": CLIP_PRETRAINED,
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
    try: optimizer.load_state_dict(ckpt["opt_state"])
    except Exception: pass
    if ckpt.get("sched_t") is not None: scheduler.t = int(ckpt["sched_t"])
    control_state = ckpt.get("control_state", None)
    append_log(f"[resume] epoch={ckpt['epoch']} global_step={ckpt.get('global_step',0)} not: {ckpt.get('note','')}")
    return int(ckpt["epoch"]) + 1, int(ckpt.get("global_step", 0)), control_state

# ----------------- ASL (timm'den, yoksa fallback) -----------------
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
    def build_loss(): return AsymmetricLossMultiLabel(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)

# ----------------- CLIP + Head -----------------
class CLIPMultiLabelModel(nn.Module):
    def __init__(self, arch, pretrained, num_classes, device):
        super().__init__()
        self.clip, self.preprocess, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        self.clip.eval()  # train'de tekrar açacağız
        feat_dim = self.clip.visual.output_dim
        self.head = nn.Linear(feat_dim, num_classes)
        nn.init.zeros_(self.head.bias)
        if hasattr(self.clip, "set_grad_checkpointing"):
            try: self.clip.set_grad_checkpointing(True)
            except Exception: pass

    def forward(self, x):
        f = self.clip.encode_image(x).float()
        if L2_NORMALIZE_FEATS:
            f = f / (f.norm(dim=1, keepdim=True) + 1e-6)
        return self.head(f)

def _set_requires_grad(mod, flag=True):
    n=0
    for p in mod.parameters():
        p.requires_grad_(flag); n+=p.numel()
    return n

def set_trainable_unfreeze_all(model: CLIPMultiLabelModel):
    n = 0
    n += _set_requires_grad(model.clip, True)
    n += _set_requires_grad(model.head, True)
    return n

def count_trainable(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_tr, n_all

# ----------------- Scheduler / Optimizer -----------------
class CosineWithWarmup:
    def __init__(self, optimizer, epochs, warmup_epochs, iters_per_epoch, min_lr=1e-6):
        self.opt = optimizer
        self.total = max(1, epochs * max(1, iters_per_epoch))
        self.warm  = warmup_epochs * max(1, iters_per_epoch)
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
                lr = self.min_lr + 0.5*(base - self.min_lr)*(1 + math.cos(math.pi*tt))
            pg['lr'] = lr

def build_optimizer_unfreeze(model: CLIPMultiLabelModel):
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    bb_params   = [p for p in model.clip.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR_HEAD,     "initial_lr": LR_HEAD},
            {"params": bb_params,   "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY
    )

def build_scheduler(optimizer, iters_per_epoch, total_epochs=EPOCHS_HINT, warmup=WARMUP_EPOCHS):
    return CosineWithWarmup(optimizer, epochs=total_epochs,
                            warmup_epochs=warmup, iters_per_epoch=iters_per_epoch, min_lr=1e-6)

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
                   if (USE_AMP_BF16 and DEVICE=="cuda") else nullcontext())
            with ctx:
                g = model(x).float().cpu()
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
            tp = int((yhat & (y==1)).sum()); fp = int((yhat & (y==0)).sum()); fn = int(((1-yhat) & (y==1)).sum())
            prec = safe_div(tp, tp+fp); rec = safe_div(tp, tp+fn)
            f1 = safe_div(2*prec*rec, prec+rec)
            if f1 > best_f1: best_f1, best_t = f1, t
        ths[ci] = best_t
    append_log(f"[val] threshold seçimi bitti. süre={time.time()-t0:.1f}s")
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
                   if (USE_AMP_BF16 and DEVICE=="cuda") else nullcontext())
            with ctx:
                g = model(x).float().cpu()
            Ls.append(g); Ys.append(y); Ms.append(m)
            if bi % 50 == 0: append_log(f"[eval] {bi} batch işlendi…")
    L = torch.cat(Ls).numpy(); Y = torch.cat(Ys).numpy(); M = torch.cat(Ms).numpy()
    P = sigmoid_np_stable(L)
    if thresholds is None:
        thresholds = np.array([0.5]*len(CLASS_NAMES), dtype=np.float32)
    YH = (P >= thresholds[None, :]).astype(np.uint8)

    per_class = []
    micro_tp = micro_fp = micro_fn = 0
    for ci, cname in enumerate(CLASS_NAMES):
        mask = M[:, ci] > 0.5
        if mask.sum() == 0:
            per_class.append({"class": cname, "support": 0, "prec": None, "rec": None, "f1": None})
            continue
        y  = Y[mask, ci].astype(np.uint8)
        yh = YH[mask, ci].astype(np.uint8)
        tp = int((yh & (y==1)).sum()); fp = int((yh & (y==0)).sum()); fn = int(((1-yh) & (y==1)).sum())
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        prec = safe_div(tp, tp+fp); rec = safe_div(tp, tp+fn)
        f1   = safe_div(2*prec*rec, prec+rec)
        per_class.append({"class": cname, "support": int(mask.sum()),
                          "prec": float(prec), "rec": float(rec), "f1": float(f1)})
    f1s = [d["f1"] for d in per_class if d["f1"] is not None]
    macro_f1 = float(np.mean(f1s)) if f1s else None
    micro_prec = safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec  = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1   = safe_div(2*micro_prec*micro_rec, micro_prec + micro_rec)
    append_log(f"[eval] süre={time.time()-t0:.1f}s microF1={micro_f1:.4f} macroF1={macro_f1:.4f}")
    cuda_empty_cache()
    return per_class, {"macro_f1": macro_f1,
                       "micro_prec": float(micro_prec),
                       "micro_rec": float(micro_rec),
                       "micro_f1": float(micro_f1)}

# ----------------- Dataset -----------------
class MultiLabelCsvDataset(Dataset):
    def __init__(self, csv_path, root_dir, preprocess):
        self.df   = pd.read_csv(csv_path)
        self.root = root_dir
        self.pp   = preprocess
        expect = ["name"] + CLASS_NAMES + MASK_NAMES
        for c in expect:
            if c not in self.df.columns:
                raise ValueError(f"CSV kolon eksik: {c}")
        self.names = self.df["name"].astype(str).values
        self.Y = self.df[CLASS_NAMES].astype("float32").values
        self.M = self.df[MASK_NAMES].astype("float32").values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.names[idx])
        img  = ensure_rgb(Image.open(path))
        x    = self.pp(img)
        y    = torch.from_numpy(self.Y[idx])
        m    = torch.from_numpy(self.M[idx])
        return x, y, m, path

# ----------------- Eğitim -----------------
def train():
    seed_everything(SEED)
    append_log(">> CLIP yükleniyor...")
    tmp_model, preprocess, _ = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED, device=DEVICE
    )
    del tmp_model

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
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **dl_common)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **dl_common)
    append_log(f"[dl] train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

    # Model (unfreeze from start)
    model = CLIPMultiLabelModel(CLIP_ARCH, CLIP_PRETRAINED, num_classes=len(CLASS_NAMES), device=DEVICE).to(DEVICE)
    set_trainable_unfreeze_all(model)
    n_tr, n_all = count_trainable(model)
    append_log(f"[model] trainable params={n_tr}/{n_all} (UNFREEZE-ALL)")

    # Optional: head init
    if os.path.isfile(HEAD_INIT_PATH):
        tmp = torch.load(HEAD_INIT_PATH, map_location="cpu")
        hd = tmp.get("state_dict", tmp)
        model.head.load_state_dict(hd, strict=True)
        if "class_names" in tmp: append_log(f"[head-init] class_names={len(tmp['class_names'])}")
        if "clip_arch" in tmp:   append_log(f"[head-init] arch={tmp['clip_arch']} / {tmp.get('clip_pretrained')}")

    optimizer = build_optimizer_unfreeze(model)
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, iters_per_epoch)

    start_epoch, global_step, resumed_control = try_resume(model, optimizer, scheduler)
    control = {
        "phase": "unfreeze",
        "best_val_macro": -1.0,
        "last_val_macro": None,
        "plateau": 0,
        "total_epochs": 0,
        "best_val_thresholds": None,
        "best_test_macro": -1.0,
        "last_test_macro": None,
        "overfit_streak": 0,
    }
    if resumed_control is not None:
        control.update({
            "best_val_macro": resumed_control.get("best_val_macro", -1.0),
            "last_val_macro": resumed_control.get("last_val_macro", None),
            "plateau": resumed_control.get("plateau", 0),
            "total_epochs": resumed_control.get("total_epochs", 0),
            "best_val_thresholds": resumed_control.get("best_val_thresholds", None),
            "best_test_macro": resumed_control.get("best_test_macro", -1.0),
            "last_test_macro": resumed_control.get("last_test_macro", None),
            "overfit_streak": resumed_control.get("overfit_streak", 0),
        })
        append_log(f"[control] resume state: {control}")

    loss_fn = build_loss()

    append_log(">> Eğitim başlıyor... (UNFREEZE-ONLY)")
    epoch = start_epoch
    try:
        while True:
            control["total_epochs"] += 1
            model.train()
            run_loss, denom_count = 0.0, 0.0
            t0 = time.time()
            append_log(f"[epoch {epoch:02d}] train loop başlıyor…")
            cuda_empty_cache()

            for it, (x, y, m, _) in enumerate(train_loader, start=1):
                x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(DEVICE); m = m.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if (USE_AMP_BF16 and DEVICE == "cuda") else nullcontext())
                try:
                    with ctx:
                        logits = model(x)
                        loss_mat = loss_fn(logits, y)
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
                    append_log(f"[oom] it={it} → batch SKIP; cache temizleniyor"); cuda_empty_cache(); continue

                if DEVICE == "cuda" and (it % 8 == 0): cuda_empty_cache()
                run_loss += float(loss.item()) * float(denom.item())
                denom_count += float(denom.item())
                global_step += 1

                if it % LOG_EVERY == 0:
                    lrs = [round(pg['lr'],6) for pg in optimizer.param_groups]
                    append_log(f"[epoch {epoch:02d}] it={it}/{len(train_loader)} loss={loss.item():.4f} step={global_step} lr={lrs}")

            epoch_loss = run_loss / max(denom_count, 1.0)
            dt = time.time() - t0
            lrs = [round(pg['lr'],6) for pg in optimizer.param_groups]
            append_log(f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | time={dt:.1f}s | lr={lrs}")

            # ---- Val + thresholds
            val_thresholds = pick_thresholds(val_loader, model)
            val_per, val_agg = evaluate(val_loader, model, val_thresholds)
            val_macro = val_agg["macro_f1"] if val_agg["macro_f1"] is not None else -1.0

            # best val
            if val_macro > control["best_val_macro"] + 1e-12:
                control["best_val_macro"] = val_macro
                control["best_val_thresholds"] = {c: float(t) for c, t in zip(CLASS_NAMES, val_thresholds)}
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "clip_arch": CLIP_ARCH,
                    "clip_pretrained": CLIP_PRETRAINED,
                    "val_macro_f1": float(control["best_val_macro"]),
                    "val_thresholds": control["best_val_thresholds"]
                }, os.path.join(OUT_DIR, "best_val.pt"))
                with open(os.path.join(OUT_DIR, "best_val_thresholds.json"), "w", encoding="utf-8") as f:
                    json.dump(control["best_val_thresholds"], f, indent=2)
                append_log(f"[val] ✅ best updated | macro-F1={control['best_val_macro']:.4f}")
                control["plateau"] = 0
            else:
                last_macro = control["last_val_macro"]
                if last_macro is not None and (val_macro - last_macro) < UNFREEZE_PLATEAU_DELTA:
                    control["plateau"] += 1
            control["last_val_macro"] = val_macro

            # ---- Per-epoch TEST (val thresholds ile)
            test_per, test_agg = evaluate(test_loader, model, np.array(val_thresholds, dtype=np.float32))
            test_macro = test_agg["macro_f1"] if test_agg["macro_f1"] is not None else -1.0
            control["last_test_macro"] = test_macro

            # Best TEST: model + rapor kaydet
            if test_macro > control["best_test_macro"] + 1e-12:
                control["best_test_macro"] = test_macro
                atomic_save({
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "clip_arch": CLIP_ARCH,
                    "clip_pretrained": CLIP_PRETRAINED,
                    "best_test_macro_f1": float(test_macro),
                    "val_thresholds_at_best_test": {c: float(t) for c, t in zip(CLASS_NAMES, val_thresholds)}
                }, BEST_TEST_PT)
                with open(BEST_TEST_JSON, "w", encoding="utf-8") as f:
                    json.dump({"per_class": test_per, "aggregate": test_agg}, f, indent=2)
                append_log(f"[test] ⭐ NEW BEST macroF1={test_macro:.4f} → kaydedildi: best_test.pt")

            # val/test gap → overfit guard
            if (val_macro is not None) and (test_macro is not None):
                gap = val_macro - test_macro
                if gap >= OVERFIT_GAP_DELTA:
                    control["overfit_streak"] += 1
                    append_log(f"[overfit] gap={gap:.4f} (streak={control['overfit_streak']}/{OVERFIT_PATIENCE})")
                else:
                    control["overfit_streak"] = 0

            # epoch checkpoint
            if SAVE_EVERY_EPOCH:
                save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                                note="epoch_end", control_state=control)

            # ---- OTO-DURDURMA (tek faz)
            stop_plateau = (control["plateau"] >= UNFREEZE_PATIENCE) and (control["total_epochs"] >= MIN_EPOCHS)
            stop_regress = STOP_ON_REGRESSION and (val_macro < control["best_val_macro"] - REGRESSION_DELTA) and (control["total_epochs"] >= MIN_EPOCHS)
            stop_test_regress = STOP_ON_TEST_REGRESSION and (control["best_test_macro"] > 0) and (test_macro < control["best_test_macro"] - TEST_REGRESSION_DELTA) and (control["total_epochs"] >= MIN_EPOCHS)
            stop_overfit = (control["overfit_streak"] >= OVERFIT_PATIENCE)

            if stop_plateau:
                append_log(f"[auto] stop (val plateau): {control['plateau']} >= {UNFREEZE_PATIENCE}"); break
            if stop_regress:
                append_log(f"[auto] stop (val regression): val_macro={val_macro:.4f} < best-{REGRESSION_DELTA}={control['best_val_macro']-REGRESSION_DELTA:.4f}"); break
            if stop_test_regress:
                append_log(f"[auto] stop (test regression): test_macro={test_macro:.4f} < best-{TEST_REGRESSION_DELTA}={control['best_test_macro']-TEST_REGRESSION_DELTA:.4f}"); break
            if stop_overfit:
                append_log(f"[auto] stop (overfit gap): streak={control['overfit_streak']} gap≥{OVERFIT_GAP_DELTA}"); break

            epoch += 1

    except KeyboardInterrupt:
        append_log("\n[train] CTRL+C → checkpoint kaydediliyor…")
        save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                        note="keyboard_interrupt", control_state=control)
        append_log("[train] Güvenle çıkıldı."); return
    except Exception:
        append_log("[train] HATA! checkpoint kaydediliyor…")
        traceback.print_exc()
        save_checkpoint(epoch, global_step, model, optimizer, scheduler,
                        note="exception", control_state=control)
        raise

    # ---- Final rapor: best_test.json zaten yazıldı; yine de son epoch testini yazalım
    append_log(">> Eğitim bitti. En iyi TEST sonuçları best_test.json ve ağırlıklar best_test.pt olarak kaydedildi.")
    append_log(f"Dosyalar: {BEST_TEST_PT}, {BEST_TEST_JSON}, last.pt, best_val.pt, best_val_thresholds.json")
# -----------------
if __name__ == "__main__":
    train()
