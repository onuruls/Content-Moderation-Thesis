# src/models/camie_tagger.py
import json, numpy as np
from typing import List, Optional, Dict
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PAD_COLOR_RGB = (124, 116, 104)
_CHUNK = 16

def _pil_to_chw_float_tensor(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB") if img.mode != "RGB" else img
    w, h = img.size
    ar = w / h if h else 1.0
    if ar > 1.0:
        new_w = size; new_h = int(round(new_w / ar))
    else:
        new_h = size; new_w = int(round(new_h * ar))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), _PAD_COLOR_RGB)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC->CHW
    return np.ascontiguousarray(x, dtype=np.float32)

def _build_rating_index(idx_to_tag: Dict[int, str], tag_to_cat: Dict[str, str]) -> Dict[str, int]:
    idx = {"general": None, "sensitive": None, "questionable": None, "explicit": None}

    # 1) Prefer explicit category map if present
    for i, name in idx_to_tag.items():
        cat = tag_to_cat.get(name, "").lower()
        if cat == "rating":
            low = name.lower()
            if "general" in low or low.endswith(":general") or low == "general":
                idx["general"] = i
            elif "sensitive" in low:
                idx["sensitive"] = i
            elif "questionable" in low:
                idx["questionable"] = i
            elif "explicit" in low:
                idx["explicit"] = i

    # 2) Fallback: infer by tag name patterns
    def _maybe_set(key, pred):
        if idx[key] is None:
            for i, name in idx_to_tag.items():
                if pred(name.lower()):
                    idx[key] = i
                    break

    _maybe_set("general",      lambda s: s == "general" or s.endswith(":general"))
    _maybe_set("sensitive",    lambda s: "sensitive" in s)
    _maybe_set("questionable", lambda s: "questionable" in s or s.endswith(":q"))
    _maybe_set("explicit",     lambda s: "explicit" in s)

    # 3) If still missing anything, we’ll warn and treat p_unsafe as 0.0 for safety.
    return idx

class CamieTaggerV2ZS:
    """
    Category-agnostic unsafe score (like WD/Animetimm):
    p_unsafe = max(prob[sensitive], prob[questionable], prob[explicit]).
    """
    def __init__(self, repo_id: str = "Camais03/camie-tagger-v2", img_size: Optional[int] = None):
        self.repo_id = repo_id

        onnx_path = hf_hub_download(repo_id=repo_id, filename="camie-tagger-v2.onnx")
        meta_path = hf_hub_download(repo_id=repo_id, filename="camie-tagger-v2-metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        try:
            self.img_size = int(meta.get("model_info", {}).get("img_size", 512))
        except Exception:
            self.img_size = img_size or 512

        tag_map = meta["dataset_info"]["tag_mapping"]
        self.idx_to_tag = {int(k): v for k, v in tag_map["idx_to_tag"].items()}
        self.tag_to_cat = tag_map.get("tag_to_category", {})

        self.rating_idx = _build_rating_index(self.idx_to_tag, self.tag_to_cat)
        self.unsafe_indices = [i for k, i in self.rating_idx.items() if k != "general" and i is not None]
        if not self.unsafe_indices:
            print("[camie] WARNING: rating indices not resolved; p_unsafe will default to 0.")

        # Simple, stable CUDA session (Windows-friendly)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0

        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"do_copy_in_default_stream": "1",
                                 "cudnn_conv_use_max_workspace": "1",
                                 "tunable_op_enable": "1",
                                 "tunable_op_tuning_enable": "0"}, {}]
        else:
            providers = ["CPUExecutionProvider"]; provider_options = [{}]
            print("[camie] WARNING: CUDAExecutionProvider not available. Falling back to CPU.")

        self.session = ort.InferenceSession(
            onnx_path, sess_options=so,
            providers=providers, provider_options=provider_options
        )
        self.input_name = self.session.get_inputs()[0].name
        outs = self.session.get_outputs()
        self._out_name = outs[1].name if len(outs) >= 2 else outs[0].name
        print(f"[camie] Active providers: {self.session.get_providers()}")

    def _forward(self, batch_x: np.ndarray) -> np.ndarray:
        batch_x = np.require(batch_x, dtype=np.float32, requirements=["C"])
        logits = self.session.run([self._out_name], {self.input_name: batch_x})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs

    def _unsafe_from_probs(self, probs_row: np.ndarray) -> float:
        if not self.unsafe_indices:
            return 0.0
        vals = [float(probs_row[i]) for i in self.unsafe_indices if i < probs_row.shape[-1]]
        return float(max(vals)) if vals else 0.0

    # NOTE: category is ignored intentionally to match WD/Animetimm behavior.
    def predict_proba(self, pil_img: Image.Image, category: Optional[str]) -> Optional[float]:
        x = _pil_to_chw_float_tensor(pil_img, self.img_size)[None, ...]
        probs = self._forward(x)[0]
        return self._unsafe_from_probs(probs)

    def predict_proba_batch(self, pil_list: List[Image.Image], categories: Optional[List[str]] = None) -> List[float]:
        xs = [_pil_to_chw_float_tensor(im, self.img_size) for im in pil_list]
        out = []
        for i in range(0, len(xs), _CHUNK):
            chunk = np.stack(xs[i:i+_CHUNK], axis=0)
            probs = self._forward(chunk)
            out.extend([self._unsafe_from_probs(row) for row in probs])
        return out
