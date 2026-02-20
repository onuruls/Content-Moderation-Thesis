import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, brier_score_loss

def counts_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn

def metrics_from_counts(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = (prec + rec)
    f1 = (2.0 * prec * rec / denom) if denom > 0.0 else 0.0
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1)}

def binary_report(y_true, y_pred):
    tp, fp, tn, fn = counts_from_arrays(y_true, y_pred)
    m = metrics_from_counts(tp, fp, tn, fn)
    m.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": tp + fp + tn + fn})
    return m

def pr_auc(y_true, y_scores):
    if len({int(v) for v in y_true}) < 2: return 0.0
    return float(average_precision_score(y_true, y_scores))

def roc_auc(y_true, y_scores):
    try:
        if len({int(v) for v in y_true}) < 2: return 0.0
        return float(roc_auc_score(y_true, y_scores))
    except Exception: return 0.0

def recall_at_fpr(y_true, y_scores, target_fpr=0.01):
    if len({int(v) for v in y_true}) < 2: return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = fpr <= float(target_fpr)
    return float(np.max(tpr[mask])) if np.any(mask) else 0.0

def fpr_at_recall(y_true, y_scores, target_recall=0.90):
    if len({int(v) for v in y_true}) < 2: return 1.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = tpr >= float(target_recall)
    return float(np.min(fpr[mask])) if np.any(mask) else 1.0

def brier(y_true, y_scores):
    return float(brier_score_loss(y_true, y_scores))

def ece(y_true, y_scores, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_scores, bins) - 1
    ece_val = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask): continue
        conf = float(np.mean(y_scores[mask]))
        acc  = float(np.mean(y_true[mask]))
        w    = float(np.mean(mask))
        ece_val += w * abs(acc - conf)
    return float(ece_val)
