# src/data/nudenet.py
import random
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from src.data.util import apply_slice, is_img, normalize_split_name

def load_nudenet(split: str = "test", sexy_policy: str = "exclude") -> List[Dict[str, Any]]:
	root = Path("src/data/nudenet").expanduser().resolve()
	base = normalize_split_name(split)
	base_dir = root / base
	if not base_dir.exists():
		raise FileNotFoundError(f"{base_dir} not found. Expected {root}/training|validation|testing/(nude|safe|sexy)/")

	by_class: Dict[str, List[str]] = {}
	for cls in ["nude", "safe", "sexy"]:
		d = base_dir / cls
		if d.exists():
			by_class[cls] = sorted(str(p) for p in d.rglob("*") if is_img(p))

	rows_all: List[Dict[str, Any]] = []
	for cls, paths in by_class.items():
		if cls == "safe":
			target = "safe"
		elif cls == "nude":
			target = "unsafe"
		elif cls == "sexy":
			if sexy_policy == "exclude":
				continue
			target = "unsafe" if sexy_policy == "unsafe" else "safe"
		else:
			continue
		for p in paths:
			rows_all.append({
				"image": p,
				"safety_label": target,
				"category": "sexual",
				"source": "NudeNet",
			})

	rnd = random.Random(42)
	rows_shuf = list(rows_all)
	rnd.shuffle(rows_shuf)

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
