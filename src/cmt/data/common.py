import re
from pathlib import Path
from typing import List, TypeVar, Sequence

T = TypeVar("T")

def apply_slice(items: Sequence[T], split: str) -> List[T]:
    """
    Apply slice suffix like 'train[:100]' or 'train[0.8:]'.
    """
    m = re.search(r"\[(.*)\]$", split)
    if not m:
        return list(items)
    
    n = len(items)
    sl = m.group(1).strip()

    def parse_idx(x: str, default: int) -> int:
        if not x: return default
        if x.endswith("%"):
            return int(float(x[:-1]) * 0.01 * n)
        return int(x)

    if ":" in sl:
        parts = sl.split(":", 1)
        start = parse_idx(parts[0], 0)
        end = parse_idx(parts[1], n)
        return list(items[start:end])
    else:
        start = parse_idx(sl, 0)
        return list(items[start:])

def is_img(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

def normalize_split_name(split: str) -> str:
    """
    Normalize 'train', 'val', 'test' to dataset specific folder names if needed.
    Returns: 'training', 'validation', 'testing' as standard directory names.
    """
    # Remove slice if present for base name
    base = re.split(r"\[", split)[0].strip().lower()
    
    mapping = {
        "train": "training",
        "training": "training",
        "val": "validation",
        "validation": "validation",
        "test": "testing",
        "testing": "testing",
    }
    
    return mapping.get(base, base)
