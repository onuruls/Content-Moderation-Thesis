# src/eval/ft/siglip/select_threshold_and_test.py
import json, numpy as np
from sklearn.metrics import (precision_recall_curve, classification_report,
                             confusion_matrix, average_precision_score,
                             roc_auc_score, roc_curve)
from scipy.special import expit

# ========= user settings =========
PREFIX  = "outputs/siglip_nudenet"
VAL_TAGS = ["validation_50", "validation", "val"]
TEST_TAGS = ["testing_100", "testing", "test"]
HEAD_PATH = f"{PREFIX}_linear_head_best.npz"
L2NORM = True
JSON_OUT = f"{PREFIX}_thresholds.json"
# =================================

def load_split(prefix, names):
    for n in names:
        try:
            X = np.load(f"{prefix}_{n}_X.npy").astype("float32")
            y = np.load(f"{prefix}_{n}_y.npy").astype("int64")
            return X, y, n
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of these splits found: {names}")

def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def logits_from(X, w, b, l2=False, bs=32768):
    if l2: X = l2norm_rows(X)
    z = np.empty((X.shape[0],), dtype=np.float32)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs]
        z[i:i+bs] = (xb @ w).ravel() + b
    return z

def ece(y_true, p, n_bins=15):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(idx): continue
        total += idx.mean() * abs(p[idx].mean() - y[idx].mean())
    return float(total)

def recall_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    i = np.where(fpr <= target_fpr)[0]
    return float(tpr[i].max()) if len(i) else 0.0

def fpr_at_recall(y_true, scores, target_recall):
    fpr, tpr, _ = roc_curve(y_true, scores)
    j = np.where(tpr >= target_recall)[0]
    return float(fpr[j].min()) if len(j) else 1.0

# ---- load head ----
wb = np.load(HEAD_PATH)
w = wb["w"].astype("float32").reshape(-1, 1)   # (D,1)
b = float(wb["b"].reshape(()))                 # scalar

# ---- validation ----
Xv, yv, vname = load_split(PREFIX, VAL_TAGS)
zv = logits_from(Xv, w, b, l2=L2NORM)
# precision_recall_curve works with any score; here we pass logits to keep thresholds in logit space
prec, rec, thr_logit = precision_recall_curve(yv, zv)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1[:-1]))         # align with thr_logit (length-1)
t_f1_logit = float(thr_logit[best_idx])
print(f"[VAL {vname}] F1-max thr_logit = {t_f1_logit:.6f}  P={prec[best_idx]:.4f}  R={rec[best_idx]:.4f}  F1={f1[best_idx]:.4f}")

# Highest threshold achieving recall >= target_R
target_R = 0.95
ok = np.where(rec[:-1] >= target_R)[0]
if len(ok):
    # PR curve thresholds are sorted in decreasing score order; first index = highest threshold
    idx = int(ok[0])
    t_rec_logit = float(thr_logit[idx])
    print(f"[VAL {vname}] R>={target_R} thr_logit = {t_rec_logit:.6f}  P={prec[idx]:.4f}  R={rec[idx]:.4f}  F1={f1[idx]:.4f}")
else:
    t_rec_logit = t_f1_logit
    print(f"[VAL {vname}] R>={target_R} not reachable; using F1-max threshold.")

# ---- test ----
Xt, yt, tname = load_split(PREFIX, TEST_TAGS)
zt = logits_from(Xt, w, b, l2=L2NORM)

def full_report(y, z, thr):
    pred = (z >= thr).astype(int)
    rep = classification_report(y, pred, digits=4)
    cm  = confusion_matrix(y, pred)
    p   = expit(np.clip(z, -50, 50))
    prA = float(average_precision_score(y, p))
    try: roA = float(roc_auc_score(y, p))
    except: roA = 0.0
    e   = ece(y, p, n_bins=15)
    r1  = recall_at_fpr(y, p, 0.01)
    r5  = recall_at_fpr(y, p, 0.05)
    f90 = fpr_at_recall(y, p, 0.90)
    f95 = fpr_at_recall(y, p, 0.95)
    print(rep)
    print("Confusion:\n", cm)
    print(json.dumps({
        "pr_auc": prA, "roc_auc": roA, "ece_15": e,
        "recall_at_1pct_fpr": r1, "recall_at_5pct_fpr": r5,
        "fpr_at_90pct_recall": f90, "fpr_at_95pct_recall": f95
    }, indent=2))

print(f"\n[TEST {tname}] @ thr_logit(F1-max) = {t_f1_logit:.6f}")
full_report(yt, zt, t_f1_logit)

print(f"\n[TEST {tname}] @ thr_logit(R>={target_R}) = {t_rec_logit:.6f}")
full_report(yt, zt, t_rec_logit)

# save thresholds (for runner/inference integration if needed)
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "thr_logit_f1max": float(t_f1_logit),
        "thr_logit_r95": float(t_rec_logit),
        "val_split": vname,
        "test_split": tname,
        "l2norm": bool(L2NORM)
    }, f, indent=2)
print(f"\nSaved thresholds -> {JSON_OUT}")
