# src/data/unsafebench.py
from datasets import load_dataset
from PIL import Image

def load_unsafebench(split="test"):
	ds = load_dataset("yiting/UnsafeBench", split=split)
	return ds

def iter_images(ds, filtered_cats=None):
	for ex in ds:
		if filtered_cats:
			if (ex.get("category") or "").lower() not in filtered_cats:
				continue
		img = ex["image"] if isinstance(ex["image"], Image.Image) else Image.open(ex["image"])
		yield {
			"image": img.convert("RGB"),
			"label": 1 if ex["safety_label"].lower()=="unsafe" else 0,
			"category": ex["category"].lower(),
			"source": ex["source"]
		}
