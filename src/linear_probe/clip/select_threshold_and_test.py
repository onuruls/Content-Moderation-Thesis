import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from scipy.special import expit

prefix = "outputs/clip_vitl14_nudenet"

wb = np.load(f"{prefix}_linear_head_best.npz")
w, b = wb["w"], wb["b"]

def load_split(prefix, names):
    for n in names:
        try:
            X = np.load(f"{prefix}_{n}_X.npy").astype("float32")
            y = np.load(f"{prefix}_{n}_y.npy").astype("int64")
            return X, y, n
        except FileNotFoundError:
            pass
    raise FileNotFoundError(names)

Xv, yv, vname = load_split(prefix, ["validation", "validation_50", "val"])
logit_v = (Xv @ w + b).astype("float64").ravel()

prec, rec, thr_logit = precision_recall_curve(yv, logit_v)
f1 = 2*prec*rec/(prec+rec+1e-12)

best_idx = int(np.nanargmax(f1[:-1]))
t_f1_logit = float(thr_logit[best_idx])
print(f"[VAL {vname}] F1-max thr_logit = {t_f1_logit:.4f}, P={prec[best_idx]:.4f}, R={rec[best_idx]:.4f}, F1={f1[best_idx]:.4f}")

target_R = 0.95
ok = np.where(rec[:-1] >= target_R)[0]
if len(ok):
    idx = int(ok[-1])
    t_rec_logit = float(thr_logit[idx])
    print(f"[VAL {vname}] R>={target_R} thr_logit = {t_rec_logit:.4f}, P={prec[idx]:.4f}, R={rec[idx]:.4f}, F1={f1[idx]:.4f}")
else:
    t_rec_logit = t_f1_logit
    print(f"[VAL {vname}] R>={target_R} sağlanamadı, F1-max kullanılacak.")

# ---- Test ----
Xt, yt, tname = load_split(prefix, ["testing_100", "test"])
logit_t = (Xt @ w + b).astype("float64").ravel()

for tag, thr_use in [("F1max", t_f1_logit), (f"R>={target_R}", t_rec_logit)]:
    pred = (logit_t >= thr_use).astype(int)
    print(f"\n[TEST {tname}] thr_logit={thr_use:.4f}  ({tag})")
    print(classification_report(yt, pred, digits=4))
    print("Confusion:\n", confusion_matrix(yt, pred))

p_v = expit(np.clip(logit_v, -50, 50))
p_t = expit(np.clip(logit_t, -50, 50))
