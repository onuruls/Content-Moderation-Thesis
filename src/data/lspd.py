# src/data/lspd.py
import random
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from src.data.util import apply_slice, is_img, normalize_split_name

def load_lspd(
    split: str = "test",
    domain: str = "both",           # "anime" | "normal" | "both"
) -> List[Dict[str, Any]]:
    root = Path("src/data/lspd").expanduser().resolve()
    assert domain in {"anime", "normal", "both"}, f"domain invalid: {domain}"

    domains = ["anime", "normal"] if domain == "both" else [domain]

    rows_all: List[Dict[str, Any]] = []
    for d in domains:
        for cls in ["safe", "unsafe"]:
            dpath = root / d / cls
            if not dpath.exists():
                continue
            paths = sorted(str(p) for p in dpath.rglob("*") if is_img(p))
            target = "unsafe" if cls == "unsafe" else "safe"
            for p in paths:
                rows_all.append({
                    "image": p,
                    "safety_label": target,
                    "category": "sexual",
                    "source": f"LSPD/{d}",
                })

    rnd = random.Random(42)
    rows_shuf = list(rows_all); rnd.shuffle(rows_shuf)
    rows = apply_slice(rows_shuf, split)

    if len({r["safety_label"] for r in rows}) < 2 and len(rows) > 0:
        pos_all = [r for r in rows_all if r["safety_label"] == "unsafe"]
        neg_all = [r for r in rows_all if r["safety_label"] == "safe"]
        rnd.shuffle(pos_all); rnd.shuffle(neg_all)
        target_total = len(rows)
        k_each = min(len(pos_all), len(neg_all), max(1, target_total // 2))
        balanced = pos_all[:k_each] + neg_all[:k_each]
        remaining = max(0, target_total - len(balanced))
        if remaining > 0:
            fill = (pos_all[k_each:] + neg_all[k_each:])
            rnd.shuffle(fill)
            balanced += fill[:remaining]
        rnd.shuffle(balanced)
        rows = balanced

    return rows

def iter_images(ds, filtered_cats=None):
    for ex in ds:
        if filtered_cats and (ex.get("category") or "").lower() not in filtered_cats:
            continue
        with Image.open(ex["image"]) as img:
            img = img.convert("RGB")
        yield {
            "image": img,
            "label": 1 if ex["safety_label"].lower() == "unsafe" else 0,
            "category": ex["category"].lower(),
            "source": ex["source"],
        }
