import random
from pathlib import Path
from typing import List, Dict, Any, Generator
from PIL import Image
from cmt.data.common import apply_slice, is_img, normalize_split_name
from cmt.utils.image_utils import ensure_rgb


def iter_nudenet(split: str = "test", sexy_policy: str = "exclude", root: str = "src/data/nudenet") -> Generator[Dict[str, Any], None, None]:
    root_path = Path(root).expanduser().resolve()
    base = normalize_split_name(split)
    base_dir = root_path / base
    
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} not found. Expected {root_path}/training|validation|testing/(nude|safe|sexy)/")

    by_class: Dict[str, List[Path]] = {}
    for cls in ["nude", "safe", "sexy"]:
        d = base_dir / cls
        if d.exists():
            by_class[cls] = sorted(p for p in d.rglob("*") if is_img(p))

    rows: List[Dict[str, Any]] = []
    for cls, paths in by_class.items():
        if cls == "safe": 
            target = "safe"
        elif cls == "nude": 
            target = "unsafe"
        elif cls == "sexy":
            if sexy_policy == "exclude": continue
            target = "unsafe" if sexy_policy == "unsafe" else "safe"
        else: continue
            
        for p in paths:
            rows.append({
                "image_path": str(p),
                "safety_label": target,
                "category": "sexual",
                "source": "NudeNet"
            })
            
    # Shuffle & Slice
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
