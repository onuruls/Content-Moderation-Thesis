import numpy as np
from sklearn.metrics import precision_recall_curve

def choose_threshold_by_f1(y_true, y_scores, clamp_lo=0.1, clamp_hi=0.9):
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    
    if len({int(v) for v in y_true}) < 2:
        return 0.5, 0.0

    p, r, thr = precision_recall_curve(y_true, y_scores)
    f1 = (2 * p * r) / (p + r + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    best_t = float(thr[best_idx])
    best_f1 = float(f1[best_idx])
    
    return float(np.clip(best_t, clamp_lo, clamp_hi)), best_f1

def align_thr_to_prev(y, s, t, tol=0.15, clamp=(0.05, 0.95)):
    """
    Adjust threshold t to match the prevalence of positive class in y,
    if the current positive rate with t differs significantly from prevalence.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if y.size == 0 or s.size == 0: return float(t)
    
    prev = float(y.mean())
    pos_rate = float((s >= t).mean())
    
    if abs(pos_rate - prev) <= tol:
        return float(np.clip(t, clamp[0], clamp[1]))
        
    q = 1.0 - prev
    t_prev = float(np.quantile(s, q))
    t_new = max(t, t_prev) if pos_rate > prev else min(t, t_prev)
    return float(np.clip(t_new, clamp[0], clamp[1]))
