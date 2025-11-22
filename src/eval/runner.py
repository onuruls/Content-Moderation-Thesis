import argparse, yaml, numpy as np, json, os, time, csv
import torch
from src.eval.metrics import (
    choose_threshold_by_f1, binary_report,
    pr_auc, roc_auc, ece, recall_at_fpr, fpr_at_recall,
)
from src.eval.progress import ProgressBar

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def _dbg_stats(y, s, t):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    return float((s >= t).mean()), float(y.mean()), int(y.size)

def _iter_batched(it, bs):
    buf = []
    for ex in it:
        buf.append(ex)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf

def filter_by_allowed_categories(batch, allowed_cats, model):
    kept = [ex for ex in batch if (allowed_cats is None or ex.get("category") in allowed_cats)]
    if not kept:
        return None
    imgs = [ex["image"] for ex in kept]
    cats = [ex.get("category") for ex in kept]
    ps = model.predict_proba_batch(imgs, cats)
    return kept, imgs, cats, ps

def _align_thr_to_prev(y, s, t, tol=0.15, clamp=(0.05, 0.95)):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if y.size == 0 or s.size == 0:
        return float(t)
    prev = float(y.mean())
    pos_rate = float((s >= t).mean())
    if abs(pos_rate - prev) <= tol:
        return float(np.clip(t, clamp[0], clamp[1]))
    q = 1.0 - prev
    t_prev = float(np.quantile(s, q))
    t_new = max(t, t_prev) if pos_rate > prev else min(t, t_prev)
    return float(np.clip(t_new, clamp[0], clamp[1]))

def build_model(cfg):
    name = cfg["model_name"].lower()
    if name == "clip":
        from src.models.clip import CLIPZS
        return CLIPZS(cfg["model_id"], cfg.get("pretrained"))
    elif name == "clip_lp":
        from src.models.clip_lp import CLIPLinearProbe
        return CLIPLinearProbe(cfg["model_id"],
                               cfg.get("pretrained"),
                               head_path=cfg["linear_head_path"],
                               l2norm=bool(cfg.get("l2norm")))
    elif cfg["model_name"] == "clip_multilabel":
        from src.models.clip_multilabel import CLIPMultiLabelTrained
        return CLIPMultiLabelTrained(cfg.get("model_path"), cfg.get("dtype"))
    elif name == "siglip":
        from src.models.siglip import SigLIPZS
        return SigLIPZS(cfg["model_id"], cfg.get("dtype"))
    elif name == "siglip_lp":
        from src.models.siglip_lp import SigLIPLinearProbe
        return SigLIPLinearProbe(cfg["model_id"], cfg.get("linear_head_path"), cfg.get("l2norm"), cfg.get("use_amp"))
    elif name == "siglip2":
        from src.models.siglip2 import SigLIP2ZS
        return SigLIP2ZS(cfg["model_id"], cfg.get("dtype"))
    elif name == "dinov3":
        from src.models.dinov3 import DINOv3TXT_ZS
        return DINOv3TXT_ZS(cfg.get("model_id"), cfg.get("model_entry"), cfg.get("dtype"))
    elif name == "florence2":
        from src.models.florence2 import Florence2ZS
        return Florence2ZS(cfg.get("model_id"), cfg.get("dtype"))
    elif cfg["model_name"] == "hysac":
        from src.models.hysac_unsafe import HySAC_ZS
        return HySAC_ZS(cfg.get("model_id"), cfg.get("dtype"))
    elif cfg["model_name"] == "wdtagger":
        from src.models.wdtagger import WDEva02TaggerZS
        return WDEva02TaggerZS(cfg.get("repo_id"), cfg.get("dtype"))
    elif cfg["model_name"] == "wdtagger_trained":
        from src.models.wdtagger_trained import WDEva02TaggerTrained
        return WDEva02TaggerTrained(cfg.get("model_path"), cfg.get("dtype"))
    elif cfg["model_name"] == "camietagger":
        from src.models.camie_tagger import CamieTaggerV2ZS
        return CamieTaggerV2ZS(cfg.get("repo_id", "Camais03/camie-tagger-v2"))
    elif cfg["model_name"] == "animetimm":
        from src.models.animetimm import AnimetimmEva02TaggerZS
        return AnimetimmEva02TaggerZS(cfg.get("repo_id"), cfg.get("dtype"))
    elif cfg["model_name"] == "animetimm_trained":
        from src.models.animetimm_trained import AnimetimmEva02Trained
        return AnimetimmEva02Trained(cfg.get("model_path"), cfg.get("dtype"))
    else:
        raise ValueError(f"unknown model_name: {cfg['model_name']}")

def evaluate(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    run_name = cfg.get("run_name", "run")
    os.makedirs("outputs", exist_ok=True)
    batch_size = int(cfg["batch_size"]) if "batch_size" in cfg else None
    thr_map = {}

    t0 = time.perf_counter()
    n_forward_calls = 0

    cuda = torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda")
    print("Using device:", "cuda" if cuda else "cpu")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        gpu_info = {
            "name": torch.cuda.get_device_name(dev),
            "total_mem_bytes": int(props.total_memory),
            "capability": f"{props.major}.{props.minor}",
        }
    else:
        gpu_info = None

    model = build_model(cfg)
    model_name_str = str(cfg.get("model_name")).lower()
    print("Model:", model_name_str)
    allowed_cats = None
    supports_batch = hasattr(model, "predict_proba_batch")
    if batch_size is not None and supports_batch:
        print(f"Batching enabled: batch_size={batch_size}")

    static_thr = False
    if "static_thr" in cfg:
        static_thr = True
        glob_thr = float(cfg["static_thr"])
        print(f"Using static threshold: {glob_thr}")

    dataset = cfg.get("dataset").lower()
    print("Dataset:", dataset)
    val_split = cfg.get("val_split")
    sexy_policy = cfg.get("sexy_policy") if dataset == "nudenet" else None
    lspd_domain = cfg.get("lspd_domain")

    if dataset == "unsafebench":
        from src.data.unsafebench import load_unsafebench, iter_images as iter_images_ub
        if not static_thr:
            ds_tr = load_unsafebench(val_split)
            validation_set = iter_images_ub(ds_tr, allowed_cats)
    elif dataset == "nudenet":
        from src.data.nudenet import load_nudenet, iter_images as iter_images_nudenet
        if not static_thr:
            ds_tr = load_nudenet(val_split, sexy_policy)
            validation_set = iter_images_nudenet(ds_tr)
    elif dataset == "lspd":
        from src.data.lspd import load_lspd, iter_images as iter_images_lspd
        if not static_thr:
            ds_tr = load_lspd(val_split, lspd_domain)
            validation_set = iter_images_lspd(ds_tr)
    elif dataset == "internal":
        from src.data.internal import load_internal, iter_images as iter_images_internal
        if not static_thr:
            ds_tr = load_internal(val_split)
            validation_set = iter_images_internal(ds_tr)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    if not static_thr:
        yv, sv = [], []
        yv_by_cat, sv_by_cat = {}, {}

        val_total = len(ds_tr)
        pb_val = ProgressBar(val_total, desc="[val] scoring")

        if batch_size is not None and supports_batch:
            processed = 0
            for batch in _iter_batched(validation_set, batch_size):
                if dataset == "nudenet":
                    kept = list(batch)
                    if not kept:
                        continue
                    imgs = [ex["image"] for ex in kept]
                    ps = model.predict_proba_batch(imgs, ["sexual"] * len(kept))
                    for ex_i, p in zip(kept, ps):
                        yv.append(ex_i["label"]); sv.append(float(p))
                        yv_by_cat.setdefault("sexual", []).append(ex_i["label"])
                        sv_by_cat.setdefault("sexual", []).append(float(p))
                    processed += len(kept)
                    pb_val.update(len(kept))
                else:
                    ret = filter_by_allowed_categories(batch, allowed_cats, model)
                    if ret is None:
                        continue
                    kept, imgs, cats, ps = ret
                    for ex_i, p in zip(kept, ps):
                        cat = ex_i.get("category")
                        yv.append(ex_i["label"]); sv.append(float(p))
                        if cat is not None:
                            yv_by_cat.setdefault(cat, []).append(ex_i["label"])
                            sv_by_cat.setdefault(cat, []).append(float(p))
                    processed += len(kept)
                    pb_val.update(len(kept))
            n_forward_calls += processed
        else:
            processed = 0
            for ex in validation_set:
                if dataset == "nudenet":
                    p = model.predict_proba(ex["image"], "sexual")
                    cat_for_thr = "sexual"
                else:
                    if allowed_cats is not None and ex.get("category") not in allowed_cats:
                        pb_val.update(1)
                        continue
                    p = model.predict_proba(ex["image"], ex["category"])
                    cat_for_thr = ex.get("category")
                if p is None:
                    pb_val.update(1)
                    continue
                yv.append(ex["label"]); sv.append(float(p))
                if cat_for_thr is not None:
                    yv_by_cat.setdefault(cat_for_thr, []).append(ex["label"])
                    sv_by_cat.setdefault(cat_for_thr, []).append(float(p))
                n_forward_calls += 1
                processed += 1
                pb_val.update(1)
        pb_val.close()

        if len(sv) == 0:
            glob_thr = 0.5
            thr_map = {}
        else:
            if len(set(int(v) for v in yv)) < 2:
                glob_thr, _ = 0.5, 0.0
            else:
                glob_thr, _ = choose_threshold_by_f1(np.array(yv), np.array(sv))
            print("Threshold:", glob_thr)
            thr_map = {}
            min_cat = int(cfg.get("min_cat_val_samples", 25))
            for c, ys in yv_by_cat.items():
                ss = sv_by_cat[c]
                if len(set(int(y) for y in ys)) < 2 or len(ys) < min_cat:
                    thr_map[c] = float(glob_thr)
                else:
                    t, _ = choose_threshold_by_f1(np.array(ys), np.array(ss))
                    thr_map[c] = float(t)

        if len(sv) > 0:
            align_clamp = (0.20, 0.95) if model_name_str == "hysac" else (0.05, 0.95)
            new_glob = _align_thr_to_prev(yv, sv, glob_thr, clamp=align_clamp)
            if new_glob != glob_thr:
                glob_thr = new_glob
            for c in list(thr_map.keys()):
                ys, ss = yv_by_cat.get(c, []), sv_by_cat.get(c, [])
                if len(ys) >= 10:
                    t_old = thr_map[c]
                    t_new = _align_thr_to_prev(ys, ss, t_old, clamp=align_clamp)
                    if t_new != t_old:
                        thr_map[c] = t_new

    test_split = cfg.get("test_split")
    print("Loading test split:", test_split)
    if dataset == "unsafebench":
        ds_te = load_unsafebench(test_split)
        test_set = iter_images_ub(ds_te, allowed_cats)
    elif dataset == "nudenet":
        ds_te = load_nudenet(test_split, sexy_policy)
        test_set = iter_images_nudenet(ds_te)
    elif dataset == "lspd":
        ds_te = load_lspd(test_split, lspd_domain)
        test_set = iter_images_lspd(ds_te)
    elif dataset == "internal":
        ds_te = load_internal(test_split)
        test_set = iter_images_internal(ds_te)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    print("Test set size:", len(ds_te))

    y_test_all, s_test_all = [], []
    rows = []
    detailed_path = f"outputs/{run_name}_detailed.csv"
    fieldnames = ["idx", "category", "source", "label", "p_yes", "p_no", "pred"]
    with open(detailed_path, "w", newline="", encoding="utf-8") as f_init:
        csv.DictWriter(f_init, fieldnames=fieldnames).writeheader()
        print(f"Detailed CSV: outputs/{run_name}_detailed.csv")

    with open(detailed_path, "a", newline="", encoding="utf-8") as f_det:
        w_det = csv.DictWriter(f_det, fieldnames=fieldnames)
        pending = 0
        test_total = len(ds_te)
        pb_test = ProgressBar(test_total, desc="[test] scoring")

        if batch_size is not None and supports_batch:
            i = 0
            for batch in _iter_batched(test_set, batch_size):
                if dataset == "nudenet":
                    kept = list(batch)
                    if not kept:
                        continue
                    imgs = [ex["image"] for ex in kept]
                    ps = model.predict_proba_batch(imgs, ["sexual"] * len(kept))
                    for ex_i, p in zip(kept, ps):
                        cat_true = ex_i.get("category")
                        thr = thr_map.get("sexual", float(glob_thr))
                        pred = 1 if float(p) >= thr else 0
                        row = {
                            "idx": i, "category": cat_true, "source": ex_i.get("source"),
                            "label": int(ex_i["label"]), "p_yes": float(p),
                            "p_no": float(1.0 - float(p)), "pred": int(pred),
                        }
                        y_test_all.append(int(row["label"]))
                        s_test_all.append(float(row["p_yes"]))
                        rows.append(row); w_det.writerow(row)
                        pending += 1
                        i += 1
                    pb_test.update(len(kept))
                else:
                    ret = filter_by_allowed_categories(batch, allowed_cats, model)
                    if ret is None:
                        continue
                    kept, imgs, cats, ps = ret
                    for ex_i, p in zip(kept, ps):
                        cat = ex_i.get("category")
                        thr = thr_map.get(cat, float(glob_thr))
                        pred = 1 if float(p) >= thr else 0
                        row = {
                            "idx": i, "category": cat, "source": ex_i.get("source"),
                            "label": int(ex_i["label"]), "p_yes": float(p),
                            "p_no": float(1.0 - float(p)), "pred": int(pred),
                        }
                        y_test_all.append(int(row["label"]))
                        s_test_all.append(float(row["p_yes"]))
                        rows.append(row); w_det.writerow(row)
                        pending += 1
                        i += 1
                    pb_test.update(len(kept))
            if pending:
                f_det.flush(); os.fsync(f_det.fileno())
            n_forward_calls += i
        else:
            for i, ex in enumerate(test_set):
                if dataset == "nudenet":
                    p = model.predict_proba(ex["image"], "sexual")
                    thr = thr_map.get("sexual", float(glob_thr))
                else:
                    p = model.predict_proba(ex["image"], ex["category"])
                    thr = thr_map.get(ex.get("category"), float(glob_thr))
                if p is None:
                    pb_test.update(1)
                    continue
                row = {
                    "idx": i, "category": ex.get("category"), "source": ex.get("source"),
                    "label": int(ex["label"]), "p_yes": float(p),
                    "p_no": float(1.0 - float(p)), "pred": int(1 if float(p) >= thr else 0),
                }
                y_test_all.append(int(row["label"]))
                s_test_all.append(float(row["p_yes"]))
                rows.append(row); w_det.writerow(row)
                pending += 1
                pb_test.update(1)
            if pending:
                f_det.flush(); os.fsync(f_det.fileno())
        pb_test.close()

    cats_all = {r["category"] for r in rows if r.get("category") is not None}
    cats = sorted(cats_all if allowed_cats is None else cats_all & set(allowed_cats))
    srcs = sorted({r["source"] for r in rows if r.get("source") is not None})

    def _report(filter_fn):
        yt = [r["label"] for r in rows if filter_fn(r)]
        yp = [r["pred"] for r in rows if filter_fn(r)]
        return binary_report(np.array(yt), np.array(yp)) if yt else None

    def _report_prob(y_true, y_scores):
        if not y_true:
            return None
        if len(set(int(v) for v in y_true)) < 2:
            return {"pr_auc": 0.0, "roc_auc": 0.0, "ece_15": 0.0, "brier": 0.0}
        return {
            "pr_auc": pr_auc(y_true, y_scores),
            "roc_auc": roc_auc(y_true, y_scores),
            "ece_15": ece(y_true, y_scores, n_bins=15),
        }

    results = {"overall": _report(lambda r: True)}
    for c in cats:
        rep = _report(lambda r, c=c: r["category"] == c)
        if rep is not None:
            results[f"cat::{c}"] = rep
    for s in srcs:
        rep = _report(lambda r, s=s: r["source"] == s)
        if rep is not None:
            results[f"src::{s}"] = rep

    if len(s_test_all) > 0 and len(set(int(v) for v in y_test_all)) >= 2:
        overall_probs = {
            "pr_auc": pr_auc(y_test_all, s_test_all),
            "roc_auc": roc_auc(y_test_all, s_test_all),
            "ece_15": ece(y_test_all, s_test_all, n_bins=15),
            "recall_at_1pct_fpr": recall_at_fpr(y_test_all, s_test_all, target_fpr=0.01),
            "recall_at_5pct_fpr": recall_at_fpr(y_test_all, s_test_all, target_fpr=0.05),
            "fpr_at_90pct_recall": fpr_at_recall(y_test_all, s_test_all, target_recall=0.90),
            "fpr_at_95pct_recall": fpr_at_recall(y_test_all, s_test_all, target_recall=0.95),
        }
        results["overall_probs"] = overall_probs

    for c in cats:
        ys = [r["label"] for r in rows if r.get("category") == c]
        ss = [r["p_yes"] for r in rows if r.get("category") == c]
        prob_rep = _report_prob(ys, ss)
        if prob_rep is not None:
            results[f"cat_probs::{c}"] = prob_rep

    for s in srcs:
        ys = [r["label"] for r in rows if r.get("source") == s]
        ss = [r["p_yes"] for r in rows if r.get("source") == s]
        prob_rep = _report_prob(ys, ss)
        if prob_rep is not None:
            results[f"src_probs::{s}"] = prob_rep

    duration = time.perf_counter() - t0
    if cuda:
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_alloc = peak_reserved = None

    meta = {
        "duration_sec": float(duration),
        "n_forward_calls": int(n_forward_calls),
        "gpu": gpu_info,
        "gpu_peak_alloc_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
    }

    with open(f"outputs/{run_name}.json", "w", encoding="utf-8") as f:
        json.dump({"thr": float(glob_thr),
                   "thr_by_cat": {k: float(v) for k, v in thr_map.items()},
                   "results": results, "meta": meta}, f, indent=2)

    print(f"Global threshold: {glob_thr}")
    print(f"Per-category thresholds: {len(thr_map)}")
    print(json.dumps(results.get("overall", {}), indent=2))
    print("\nSaved detailed results:")
    print(f"  outputs/{run_name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    evaluate(args.cfg)
