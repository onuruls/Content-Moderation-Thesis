# train_multilabel_clip_head_bce_then_asl_autostop.py
import os, json, time, random, math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip

# ======================== AYARLAR ========================
TRAIN_CSV = "train.csv"
VAL_CSV   = "val.csv"
TEST_CSV  = "test.csv"
IMG_ROOT  = "../data"
CLASS_NAMES = ["alcohol","drugs","weapons","gambling","nudity","sexy","smoking","violence"]
MASK_NAMES  = [f"mask_{c}" for c in CLASS_NAMES]

CLIP_ARCH       = "ViT-L-14"
CLIP_PRETRAINED = "openai"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE   = 128
MAX_EPOCHS   = 24
LR           = 1e-3
WEIGHT_DECAY = 5e-2
NUM_WORKERS  = 2
PIN_MEMORY   = True
USE_AMP      = True
L2_NORMALIZE_FEATS = True
SEED         = 42

# Oto-durdurma (TEST bazlı)
PATIENCE_EPOCHS      = 2     # test macro-F1 iyileşmezse bu kadar epoch sonra dur
TEST_REGRESSION_DELTA= 0.02  # test macro-F1 best'ten bu kadar düşerse dur

OUT_DIR = "output_clip_bce_then_asl"
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================

# ---- (opsiyonel) timm ASL, yoksa formül ile ----
try:
    from timm.loss import AsymmetricLossMultiLabel as _ASL
    class ASL_none(_ASL):
        def __init__(self, **kw): super().__init__(**kw)
    def asl_loss_matrix(logits, targets):
        return ASL_none(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)(logits, targets)
except Exception:
    def asl_loss_matrix(logits, targets, eps=1e-8):
        x_sig = torch.sigmoid(logits)
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig
        xs_neg = (xs_neg + 0.05).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=eps)) * ((1 - xs_pos) ** 0.0)
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=eps)) * (xs_pos ** 4.0)
        return -(los_pos + los_neg)  # (B,C)

# ----------------- Yardımcılar -----------------
def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def ensure_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "P" and "transparency" in im.info: im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255,255,255)); bg.paste(im, mask=im.split()[-1]); return bg
    return im.convert("RGB") if im.mode != "RGB" else im

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))
def safe_div(a,b,eps=1e-9): return a / (b + eps)

# ----------------- Dataset -----------------
class MultiLabelCsvDataset(Dataset):
    def __init__(self, csv_path, root_dir, preprocess):
        self.df   = pd.read_csv(csv_path)
        self.root = root_dir
        self.pp   = preprocess
        expect = ["name"] + CLASS_NAMES + MASK_NAMES
        for c in expect:
            if c not in self.df.columns: raise ValueError(f"CSV kolon eksik: {c}")
        self.names = self.df["name"].astype(str).values
        self.Y = self.df[CLASS_NAMES].astype("float32").values
        self.M = self.df[MASK_NAMES].astype("float32").values
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.names[idx])
        x = self.pp(ensure_rgb(Image.open(path)))
        y = torch.from_numpy(self.Y[idx]); m = torch.from_numpy(self.M[idx])
        return x, y, m, path

# ----------------- Model & Encoder -----------------
class MultiLabelHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__(); self.fc = nn.Linear(in_dim, num_classes); nn.init.zeros_(self.fc.bias)
    def forward(self, feats): return self.fc(feats)

def build_clip_encoder(arch, pretrained, device):
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    model.eval();  [p.requires_grad_(False) for p in model.parameters()]
    feat_dim = model.visual.output_dim
    return model, preprocess, feat_dim

def encode_images(encoder, x):
    with torch.inference_mode():
        if USE_AMP and DEVICE=="cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16): f = encoder.encode_image(x)
        else: f = encoder.encode_image(x)
    f = f.float()
    if L2_NORMALIZE_FEATS: f = f / (f.norm(dim=1, keepdim=True) + 1e-6)
    return f

# ----------------- pos_weight (BCE için) -----------------
def compute_pos_weight(dataset):
    Y, M = dataset.Y, dataset.M
    pos = (Y * M).sum(axis=0); neg = ((1 - Y) * M).sum(axis=0)
    pw = np.where(pos > 0, neg / np.maximum(pos, 1e-6), 1.0).astype("float32")
    pw = np.clip(pw, 0.5, 50.0)
    return torch.from_numpy(pw)

# ----------------- Eşik & Ölçüm -----------------
def pick_thresholds(val_loader, encoder, head):
    head.eval(); Ls, Ys, Ms = [], [], []
    with torch.inference_mode():
        for x,y,m,_ in val_loader:
            x=x.to(DEVICE, non_blocking=True); g=head(encode_images(encoder,x)).cpu()
            Ls.append(g); Ys.append(y); Ms.append(m)
    L = torch.cat(Ls).numpy(); Y = torch.cat(Ys).numpy(); M = torch.cat(Ms).numpy()
    P = sigmoid_np(L); C = L.shape[1]; ths = np.zeros(C, np.float32)
    grid = np.linspace(0.01, 0.99, 99)
    for ci in range(C):
        mask = M[:,ci]>0.5
        if mask.sum()==0: ths[ci]=0.5; continue
        y = Y[mask,ci].astype(np.uint8); p = P[mask,ci]; best_f1=0.0; best_t=0.5
        for t in grid:
            yh=(p>=t).astype(np.uint8)
            tp=int((yh & (y==1)).sum()); fp=int((yh & (y==0)).sum()); fn=int(((1-yh)&(y==1)).sum())
            prec=safe_div(tp,tp+fp); rec=safe_div(tp,tp+fn); f1=safe_div(2*prec*rec,prec+rec)
            if f1>best_f1: best_f1, best_t = f1, t
        ths[ci]=best_t
    return ths

def evaluate(loader, encoder, head, thresholds=None):
    head.eval(); Ls, Ys, Ms = [], [], []
    with torch.inference_mode():
        for x,y,m,_ in loader:
            x=x.to(DEVICE, non_blocking=True); g=head(encode_images(encoder,x)).cpu()
            Ls.append(g); Ys.append(y); Ms.append(m)
    L = torch.cat(Ls).numpy(); Y = torch.cat(Ys).numpy(); M = torch.cat(Ms).numpy()
    P = sigmoid_np(L)
    if thresholds is None: thresholds = np.array([0.5]*len(CLASS_NAMES), np.float32)
    YH = (P >= thresholds[None,:]).astype(np.uint8)

    per_class=[]; micro_tp=micro_fp=micro_fn=0
    for ci,cname in enumerate(CLASS_NAMES):
        mask=M[:,ci]>0.5
        if mask.sum()==0:
            per_class.append({"class":cname,"support":0,"prec":None,"rec":None,"f1":None}); continue
        y=Y[mask,ci].astype(np.uint8); yh=YH[mask,ci].astype(np.uint8)
        tp=int((yh&(y==1)).sum()); fp=int((yh&(y==0)).sum()); fn=int(((1-yh)&(y==1)).sum())
        micro_tp+=tp; micro_fp+=fp; micro_fn+=fn
        prec=safe_div(tp,tp+fp); rec=safe_div(tp,tp+fn); f1=safe_div(2*prec*rec,prec+rec)
        per_class.append({"class":cname,"support":int(mask.sum()),"prec":float(prec),"rec":float(rec),"f1":float(f1)})
    f1s=[d["f1"] for d in per_class if d["f1"] is not None]
    macro_f1=float(np.mean(f1s)) if f1s else None
    micro_prec=safe_div(micro_tp,micro_tp+micro_fp); micro_rec=safe_div(micro_tp,micro_tp+micro_fn)
    micro_f1=safe_div(2*micro_prec*micro_rec, micro_prec+micro_rec)
    return per_class, {"macro_f1":macro_f1,"micro_prec":float(micro_prec),"micro_rec":float(micro_rec),"micro_f1":float(micro_f1)}

# ----------------- Tek denemeyi çalıştır (BCE veya ASL) -----------------
def run_experiment(tag, encoder, preprocess, feat_dim, train_ds, val_ds, test_ds):
    exp_dir = os.path.join(OUT_DIR, tag); os.makedirs(exp_dir, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    head = MultiLabelHead(in_dim=feat_dim, num_classes=len(CLASS_NAMES)).to(DEVICE)
    if tag=="bce":
        pw = compute_pos_weight(train_ds).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
    else:  # "asl"
        loss_fn = None  # element-wise ASL'i çağrıda hesaplayacağız

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_test = -1.0; best_epoch = None; noimp = 0
    print(f"\n>> [{tag}] eğitim başlıyor...")
    for epoch in range(1, MAX_EPOCHS+1):
        head.train(); run_loss=0.0; seen=0.0; t0=time.time()
        for x,y,m,_ in train_loader:
            x=x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y=y.to(DEVICE); m=m.to(DEVICE)
            feats = encode_images(encoder, x)
            logits = head(feats)

            loss_mat = loss_fn(logits,y) if tag=="bce" else asl_loss_matrix(logits,y)  # (B,C)
            denom = m.sum()
            if denom.item()<1: continue
            loss = (loss_mat * m).sum() / denom.clamp_min(1.0)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_loss += loss.item()*float(denom.item()); seen += float(denom.item())
        print(f"[{tag}] Epoch {epoch:02d} | loss={run_loss/max(seen,1.0):.4f} | time={time.time()-t0:.1f}s")

        # val -> thresholds, sonra TEST ölç
        th = pick_thresholds(val_loader, encoder, head)
        _, test_agg = evaluate(test_loader, encoder, head, th)
        test_macro = test_agg["macro_f1"] if test_agg["macro_f1"] is not None else -1.0
        print(f"[{tag}]  test macroF1={test_macro:.4f} (best={best_test:.4f})")

        improved = test_macro > best_test + 1e-12
        if improved:
            best_test = test_macro; best_epoch = epoch; noimp = 0
            # Kaydet: best_test.pt + eşikler + rapor
            torch.save({
                "state_dict": head.state_dict(),
                "feat_dim": feat_dim,
                "class_names": CLASS_NAMES,
                "clip_arch": CLIP_ARCH,
                "clip_pretrained": CLIP_PRETRAINED
            }, os.path.join(exp_dir, "best_test.pt"))
            with open(os.path.join(exp_dir, "thresholds.json"), "w") as f:
                json.dump({c: float(t) for c,t in zip(CLASS_NAMES, th)}, f, indent=2)
            with open(os.path.join(exp_dir, "class_names.json"), "w") as f:
                json.dump(CLASS_NAMES, f, indent=2)
            with open(os.path.join(exp_dir, "best_test_report.json"), "w") as f:
                json.dump({"aggregate": test_agg, "epoch": best_epoch}, f, indent=2)
        else:
            noimp += 1

        if (test_macro < best_test - TEST_REGRESSION_DELTA) or (noimp >= PATIENCE_EPOCHS):
            reason = "regression" if (test_macro < best_test - TEST_REGRESSION_DELTA) else "no-improve"
            print(f"[{tag}] auto-stop: {reason} (epoch={epoch}, best_epoch={best_epoch})")
            break

    print(f"[{tag}] BİTTİ → best_epoch={best_epoch}, best_test_macro={best_test:.4f}")
    return {"best_epoch": best_epoch, "best_test_macro": best_test}

# ----------------- Ana akış -----------------
def main():
    seed_everything(SEED)
    print(">> CLIP yükleniyor...")
    encoder, preprocess, feat_dim = build_clip_encoder(CLIP_ARCH, CLIP_PRETRAINED, DEVICE)

    print(">> Datasetler...")
    train_ds = MultiLabelCsvDataset(TRAIN_CSV, IMG_ROOT, preprocess)
    val_ds   = MultiLabelCsvDataset(VAL_CSV,   IMG_ROOT, preprocess)
    test_ds  = MultiLabelCsvDataset(TEST_CSV,  IMG_ROOT, preprocess)

    # 1) BCE denemesi
    res_bce = run_experiment("bce", encoder, preprocess, feat_dim, train_ds, val_ds, test_ds)
    # 2) ASL denemesi (başlık sıfırdan)
    res_asl = run_experiment("asl", encoder, preprocess, feat_dim, train_ds, val_ds, test_ds)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"bce": res_bce, "asl": res_asl}, f, indent=2)
    print("\nÖZET:", {"bce": res_bce, "asl": res_asl})
    print(f"\nÇıktılar:\n- {OUT_DIR}/bce/best_test.pt (+ thresholds.json, best_test_report.json)\n- {OUT_DIR}/asl/best_test.pt (+ thresholds.json, best_test_report.json)")

if __name__ == "__main__":
    main()
