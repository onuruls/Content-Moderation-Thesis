import random
from pathlib import Path
from typing import List, Dict, Any, Generator
from PIL import Image
from datasets import load_dataset
from cmt.data.common import apply_slice, normalize_split_name
from cmt.utils.image_utils import ensure_rgb

def iter_internal(split: str = "test", root: str = "src/data/internal") -> Generator[Dict[str, Any], None, None]:
    base_split = normalize_split_name(split)
    
    # HF dataset expects 'train', 'validation', 'test'
    hf_split = {"training": "train", "testing": "test"}.get(base_split, base_split)
    
    # The underlying split names for the CSV files in HF repo are train, val, test
    csv_split_name = "val" if hf_split == "validation" else hf_split
    
    try:
        # Load the images directly from Hugging Face dataset
        ds_images = load_dataset("onullusoy/harmful-contents", split=hf_split)
        
        # Load the specific CSV for this split using HF's global csv builder
        csv_url = f"hf://datasets/onullusoy/harmful-contents/csv/{csv_split_name}.csv"
        ds_csv = load_dataset("csv", data_files={hf_split: csv_url}, split=hf_split)
    except Exception as e:
        print(f"Error loading internal dataset from Hugging Face: {e}")
        return

    # Create a mapping from image file name (e.g. "4.jpg") to its index in ds_images
    image_name_to_idx = {}
    if len(ds_images) > 0 and "image" in ds_images.column_names:
        image_col = ds_images.data["image"]
        for i in range(len(ds_images)):
            path_val = image_col[i].as_py()["path"]
            file_name = Path(path_val).name
            image_name_to_idx[file_name] = i

    rows: List[Dict[str, Any]] = []
    
    # Internal DS flattens multilabel into one-example-per-category for evaluation
    cats = [c for c in ds_csv.column_names if c.lower() != "name" and not c.lower().startswith("mask_")]
    
    for r in ds_csv:
        name = r.get("name")
        if not name or name not in image_name_to_idx:
            continue
            
        img_idx = image_name_to_idx[name]
        
        for cat in cats:
            val_str = str(r.get(cat, 0)).strip()
            try:
                val = 1 if float(val_str) >= 0.5 else 0
            except:
                val = 0
                
            rows.append({
                "image_idx": img_idx,
                "safety_label": "unsafe" if val == 1 else "safe",
                "category": cat.lower(),
                "source": "Internal",
                "filename": name
            })

    random.Random(42).shuffle(rows)
    rows = apply_slice(rows, split)
    
    for r in rows:
        try:
            img = ds_images[r["image_idx"]]["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)

            yield {
                "image": ensure_rgb(img),
                "label": 1 if r["safety_label"] == "unsafe" else 0,
                "category": r["category"],
                "source": r["source"],
                "meta": {"path": r["filename"]}
            }
        except Exception:
            # Skip missing or corrupted images
            continue
