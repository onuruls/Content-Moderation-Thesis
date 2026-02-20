from datasets import load_dataset
from typing import Generator, Dict, Any
from PIL import Image
from cmt.utils.image_utils import ensure_rgb

def iter_unsafebench(split: str = "test") -> Generator[Dict[str, Any], None, None]:
    # UnsafeBench on HF
    ds = load_dataset("yiting/UnsafeBench", split=split)
    
    for ex in ds:
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
            
        yield {
            "image": ensure_rgb(img),
            "label": 1 if ex["safety_label"].lower() == "unsafe" else 0,
            "category": (ex.get("category") or "").lower(),
            "source": ex.get("source", "UnsafeBench"),
            "meta": {}
        }
