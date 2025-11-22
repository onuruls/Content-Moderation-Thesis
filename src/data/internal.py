# src/data/internal_ds.py
import csv, random
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from src.data.util import apply_slice, normalize_split_name

def _to_int01(x):
    try:
        return 1 if float(str(x).strip()) >= 0.5 else 0
    except Exception:
        return 0

def load_internal(split: str = "test") -> List[Dict[str, Any]]:
    root = Path("src/data/internal").expanduser().resolve()
    data_dir = root / "data"

    base = normalize_split_name(split)
    csv_path = root / ("val_set.csv" if base == "validation" else "test_set.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Expected {root}/val_set.csv or test_set.csv")

    rows_all: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        fields = [c.strip() for c in reader.fieldnames]
        cats = [c.lower() for c in fields if c.lower() != "name"]

        for r in reader:
            name = r.get("name") or r.get("file") or r.get("image")
            if not name:
                continue
            img_path = str((data_dir / name).resolve())

            for cat in cats:
                v = _to_int01(r.get(cat, 0))
                target = "unsafe" if v == 1 else "safe"
                rows_all.append({
                    "image": img_path,
                    "safety_label": target,
                    "category": cat,
                    "source": "Internal",
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

def iter_images(ds: List[Dict[str, Any]], filtered_cats=None):
    for ex in ds:
        cat = (ex.get("category") or "").lower()
        if filtered_cats and cat not in filtered_cats:
            continue
        with Image.open(ex["image"]) as img:
            img = img.convert("RGB")
        yield {
            "image": img,
            "label": 1 if ex["safety_label"].lower() == "unsafe" else 0,
            "category": cat,
            "source": ex.get("source", "Internal"),
        }
