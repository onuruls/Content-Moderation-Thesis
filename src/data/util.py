import re
from pathlib import Path
from typing import List, TypeVar, Sequence

T = TypeVar("T")

def apply_slice(items: Sequence[T], split: str) -> List[T]:
    m = re.search(r"\[(.*)\]$", split)
    if not m:
        return list(items)
    n = len(items); sl = m.group(1)

    def idx(x: str, d: int) -> int:
        if not x:
            return d
        x = x.strip()
        if x.endswith("%"):
            return int(float(x[:-1]) * 0.01 * n)
        return int(x)

    if ":" in sl:
        a, b = sl.split(":")
        i, j = idx(a, 0), idx(b, n)
    else:
        i, j = idx(sl, 0), n
    i = max(0, min(i, n)); j = max(0, min(j, n))
    return list(items[i:j])

def is_img(p: Path) -> bool:
    img_types = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return p.suffix.lower() in img_types

def normalize_split_name(split: str) -> str:
    split_aliases = {
        "training": {"train"},
        "validation": {"validation"},
        "testing": {"test"},
    }
    base = re.split(r"\[", split)[0].strip().lower()
    for norm, aliases in split_aliases.items():
        if base in aliases:
            return norm
    raise ValueError(f"Unknown split '{split}'. Use train|validation|test (aliases: train/val/test).")
