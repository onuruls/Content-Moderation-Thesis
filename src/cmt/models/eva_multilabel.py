from __future__ import annotations
from typing import List
import torch
import gzip, csv, io
from huggingface_hub import hf_hub_download
from PIL import Image
import timm
from timm.data import resolve_model_data_config, create_transform

from cmt.utils.torch_utils import get_device, dtype_from_str
from cmt.utils.image_utils import ensure_rgb
from cmt.models.base import Model
import numpy as np

def _load_tag_list(repo_id: str) -> list[str]:
    candidates = ["selected_tags.csv", "selected_tags.csv.gz"]
    last_err = None
    for fname in candidates:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname)
            if fname.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    text = f.read()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

            buf = io.StringIO(text)
            reader = csv.DictReader(buf)
            if reader.fieldnames:
                fns_lower = [fn.lower() for fn in reader.fieldnames]
                if "name" in fns_lower:
                    name_key = reader.fieldnames[fns_lower.index("name")]
                    return [row[name_key] for row in reader]

            buf.seek(0)
            rows = list(csv.reader(buf))
            header_lower = [h.lower() for h in rows[0]]
            name_idx = header_lower.index("name") if "name" in header_lower else 1
            return [r[name_idx] for r in rows[1:]]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not resolve tag list from {candidates}. Last error: {last_err}")

class EvaMultiLabel(Model):
    def __init__(self, repo_id: str, dtype: str = "float16"):
        self.device = get_device()
        self.dtype = dtype_from_str(dtype)
        
        print(f"Loading EVA Tagger {repo_id}...")
        self.model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).to(self.device).eval()
        self.model = self.model.to(self.dtype)
        self.model = self.model.to(memory_format=torch.channels_last)

        cfg = resolve_model_data_config(self.model)
        self.preprocess = create_transform(**cfg)
        
        self.tags = _load_tag_list(repo_id)
        
        # Build rating index for easy access
        self.idx_rating = {}
        for i, t in enumerate(self.tags):
            if t.startswith("rating:"):
                self.idx_rating[t.split("rating:")[1]] = i
        
        # Fallback defaults
        for name in ["general", "sensitive", "questionable", "explicit"]:
            if name not in self.idx_rating:
                try: self.idx_rating[name] = self.tags.index(f"rating:{name}")
                except: pass

    @torch.inference_mode()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        # TIMM models generally don't expose a separate "encode" that returns embedding easily 
        # for downstream linear probes without removing the head. 
        # For this class, we assume full end-to-end usage.
        raise NotImplementedError("EvaMultiLabel does not support generic encode()")

    @torch.inference_mode()
    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        # Preprocess
        tensors = []
        for img in images:
            img = ensure_rgb(img)
            tensors.append(self.preprocess(img))
        
        batch = torch.stack(tensors).to(self.device, self.dtype, non_blocking=True)
        batch = batch.to(memory_format=torch.channels_last)
        
        logits = self.model(batch)
        return logits

    @torch.inference_mode()
    def prob(self, images: List[Image.Image]) -> np.ndarray:
        logits = self.logits(images)
        probs = torch.sigmoid(logits)
        return probs.cpu().float().numpy()

class WDEva02Tagger(EvaMultiLabel):
    def __init__(self, repo_id: str = "SmilingWolf/wd-eva02-large-tagger-v3", dtype: str = "float16"):
        super().__init__(repo_id, dtype)

class AnimetimmEva02Tagger(EvaMultiLabel):
    def __init__(self, repo_id: str = "animetimm/eva02_large_patch14_448.dbv4-full", dtype: str = "float16"):
        super().__init__(repo_id, dtype)
