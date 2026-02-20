import random
from pathlib import Path
from typing import List, Dict, Any, Generator
from PIL import Image
from cmt.data.common import apply_slice, is_img
from cmt.utils.image_utils import ensure_rgb

def iter_lspd(split: str = "test", domain: str = "both", root: str = "src/data/lspd") -> Generator[Dict[str, Any], None, None]:
    root_path = Path(root).expanduser().resolve()
    
    domains = ["anime", "normal"] if domain == "both" else [domain]
    rows: List[Dict[str, Any]] = []
    
    for d in domains:
        for cls in ["safe", "unsafe"]:
            dpath = root_path / d / cls
            if not dpath.exists(): continue
            
            paths = sorted(p for p in dpath.rglob("*") if is_img(p))
            target = "unsafe" if cls == "unsafe" else "safe"
            
            for p in paths:
                rows.append({
                    "image_path": str(p),
                    "safety_label": target,
                    "category": "sexual",
                    "source": f"LSPD/{d}"
                })
                
    random.Random(42).shuffle(rows)
    rows = apply_slice(rows, split)
    
    for r in rows:
        yield {
            "image": ensure_rgb(Image.open(r["image_path"])),
            "label": 1 if r["safety_label"] == "unsafe" else 0,
            "category": r["category"],
            "source": r["source"],
            "meta": {"path": r["image_path"]}
        }
