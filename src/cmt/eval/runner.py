import time
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from cmt.models.registry import get_model
from cmt.data.registry import get_dataset
from cmt.eval.metrics import binary_report, pr_auc, roc_auc, ece, recall_at_fpr, fpr_at_recall
from cmt.eval.calibrate import choose_threshold_by_f1, align_thr_to_prev
from cmt.utils.io_utils import save_json, ensure_dir

def _batch_iter(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def eval_run(cfg: Dict[str, Any]):
    run_name = cfg.get("run_name", "run")
    ensure_dir("results")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{run_name}] Device: {device}")

    # 1. Load Model
    print(f"[{run_name}] Loading model...")
    model = get_model(cfg)

    # 2. Validation & Thresholding
    print(f"[{run_name}] Loading val set...")
    val_iter = get_dataset(cfg["dataset"], cfg.get("val_split", "val"), cfg)
    
    y_val, s_val = [], []
    y_val_cat, s_val_cat = defaultdict(list), defaultdict(list)
    
    batch_size = int(cfg.get("batch_size", 32))
    print(f"[{run_name}] Validation scoring (BS={batch_size})...")
    
    # Loop over validation
    processed = 0
    for batch in tqdm(_batch_iter(val_iter, batch_size), desc="Val", unit="img"):
        imgs = [ex["image"] for ex in batch]
        
        # Predict probabilities
        probs = model.prob(imgs)

        # Handle different output shapes and model types
        if probs.ndim == 1:
            # Binary classification (e.g. wrapper or binary head)
            p_unsafe = probs
        elif probs.ndim == 2 and probs.shape[1] == 1:
            p_unsafe = probs.flatten()
        else:
            # Multilabel: extract "unsafe" probability scalar
            if hasattr(model, "class_names"):
                # For CLIP Multi-label, take max of unsafe-related classes or just max of all
                # Assumption: Model output covers various unsafe categories.
                p_unsafe = probs.max(axis=1)
            elif hasattr(model, "idx_rating"):
                # For EVA taggers (DANbooru tags)
                # Max of explicit, questionable, sensitive
                p_exp = probs[:, model.idx_rating.get("explicit", 0)]
                p_ques = probs[:, model.idx_rating.get("questionable", 0)]
                p_sens = probs[:, model.idx_rating.get("sensitive", 0)]
                p_unsafe = np.maximum(p_exp, np.maximum(p_ques, p_sens))
            else:
                # Fallback: max probability across all classes
                p_unsafe = probs.max(axis=1)

        for i, ex in enumerate(batch):
            p = float(p_unsafe[i])
            label = int(ex["label"])
            cat = ex["category"]
            
            y_val.append(label)
            s_val.append(p)
            y_val_cat[cat].append(label)
            s_val_cat[cat].append(p)
            
        processed += len(batch)
        
        # Cleanup PIL images to save memory?
        for ex in batch: ex["image"].close()

    # Threshold selection
    print(f"[{run_name}] Selecting thresholds...")
    glob_thr, glob_f1 = choose_threshold_by_f1(y_val, s_val)
    print(f"[{run_name}] Global Val F1: {glob_f1:.4f} @ Thr: {glob_thr:.4f}")
    
    thr_map = {}
    min_samples = 25
    for cat in y_val_cat:
        ys, ss = y_val_cat[cat], s_val_cat[cat]
        if len(ys) >= min_samples and len(set(ys)) >= 2:
            t, f1 = choose_threshold_by_f1(ys, ss)
            thr_map[cat] = t
        else:
            thr_map[cat] = glob_thr
            
    # Alignment (fix distribution shift issues if enabled)
    # Default enabled
    glob_thr = align_thr_to_prev(y_val, s_val, glob_thr)
    for cat in thr_map:
        thr_map[cat] = align_thr_to_prev(y_val_cat[cat], s_val_cat[cat], thr_map[cat])

    # 3. Test Testing
    print(f"[{run_name}] Loading test set...")
    test_iter = get_dataset(cfg["dataset"], cfg.get("test_split", "test"), cfg)
    
    rows_out = []
    y_test, s_test = [], []
    
    print(f"[{run_name}] Test scoring...")
    for batch in tqdm(_batch_iter(test_iter, batch_size), desc="Test", unit="img"):
        imgs = [ex["image"] for ex in batch]
        probs = model.prob(imgs)
        
        if probs.ndim == 1:
            p_unsafe = probs
        elif probs.ndim == 2 and probs.shape[1] == 1:
            p_unsafe = probs.flatten()
        else:
            if hasattr(model, "class_names"): 
                p_unsafe = probs.max(axis=1)
            elif hasattr(model, "idx_rating"): 
                p_exp = probs[:, model.idx_rating.get("explicit", 0)]
                p_ques = probs[:, model.idx_rating.get("questionable", 0)]
                p_sens = probs[:, model.idx_rating.get("sensitive", 0)]
                p_unsafe = np.maximum(p_exp, np.maximum(p_ques, p_sens))
            else:
                p_unsafe = probs.max(axis=1)

        for i, ex in enumerate(batch):
            p = float(p_unsafe[i])
            label = int(ex["label"])
            cat = ex["category"]
            thr = thr_map.get(cat, glob_thr)
            pred = 1 if p >= thr else 0
            
            res = {
                "image": ex["meta"].get("path", "unknown"),
                "category": cat,
                "label": label,
                "prob": p,
                "pred": pred,
                "thr": thr,
                "correct": 1 if pred == label else 0
            }
            rows_out.append(res)
            y_test.append(label)
            s_test.append(p)
            
        for ex in batch: ex["image"].close()
            
    # 4. Metrics & Report
    print(f"[{run_name}] Calculating metrics...")
    
    # helper for filtering
    def get_subset(fn):
        sub = [r for r in rows_out if fn(r)]
        yt = [r["label"] for r in sub]
        yp = [r["pred"] for r in sub]
        return yt, yp
        
    metrics = {}
    
    # Overall
    m_overall = binary_report(y_test, [r["pred"] for r in rows_out])
    m_overall["roc_auc"] = roc_auc(y_test, s_test)
    m_overall["pr_auc"]  = pr_auc(y_test, s_test)
    m_overall["ece"]     = ece(y_test, s_test)
    metrics["overall"] = m_overall
    
    # Per-category
    cats = sorted(list(set(r["category"] for r in rows_out)))
    for c in cats:
        yt, yp = get_subset(lambda r: r["category"] == c)
        if not yt: continue
        m = binary_report(yt, yp)
        # Prob metrics for that cat
        sub_s = [r["prob"] for r in rows_out if r["category"] == c]
        m["roc_auc"] = roc_auc(yt, sub_s)
        m["pr_auc"] = pr_auc(yt, sub_s)
        metrics[f"cat_{c}"] = m

    # Save
    out_file = f"results/{run_name}.json"
    save_json({
        "config": cfg,
        "metrics": metrics,
        "global_threshold": glob_thr,
        "thresholds": thr_map
    }, out_file)
    
    # Save detailed CSV (optional, maybe just keys)
    print(f"[{run_name}] Done. Results saved to {out_file}")
    
    # Print summary
    print("\nInitial Results Summary:")
    print(f"  Accuracy: {m_overall['acc']:.4f}")
    print(f"  F1 Score: {m_overall['f1']:.4f}")
    print(f"  ROC AUC:  {m_overall['roc_auc']:.4f}")

    return metrics
