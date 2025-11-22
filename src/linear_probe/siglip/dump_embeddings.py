import argparse, yaml, os, re, warnings, numpy as np, torch
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageFile, UnidentifiedImageError
from transformers import AutoProcessor, SiglipModel
from src.data.nudenet import load_nudenet
from src.eval.progress import ProgressBar

# --- PIL robustness ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# --- utils ---
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def _name_of(ex: Dict[str, Any]) -> str:
    for k in ["image","path","filepath","file","image_path","source","id","uid","name"]:
        if isinstance(ex, dict) and k in ex and ex[k] is not None:
            return str(ex[k])
    return "<unknown>"

# --- top-level, picklable preprocess ---
class SigLIPPreprocess:
    def __init__(self, processor):
        ip = processor.image_processor
        if isinstance(ip.size, dict) and "shortest_edge" in ip.size:
            self.shortest = int(ip.size["shortest_edge"])
        elif isinstance(ip.size, int):
            self.shortest = int(ip.size)
        else:
            self.shortest = int(getattr(ip, "shortest_edge", 384))

        cs = getattr(ip, "crop_size", {})
        self.crop_h = int(cs.get("height", self.shortest))
        self.crop_w = int(cs.get("width",  self.shortest))

        self.mean = np.asarray(getattr(ip, "image_mean", [0.5,0.5,0.5]), dtype=np.float32)
        self.std  = np.asarray(getattr(ip, "image_std",  [0.5,0.5,0.5]), dtype=np.float32)

        try:
            self.RESAMPLE = Image.Resampling.BICUBIC
        except Exception:
            self.RESAMPLE = Image.BICUBIC

    def _resize_shortest(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) == self.shortest:
            return img
        s = float(self.shortest) / float(min(w, h))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        return img.resize((new_w, new_h), self.RESAMPLE)

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.crop_w or h < self.crop_h:
            img = img.resize((max(self.crop_w, w), max(self.crop_h, h)), self.RESAMPLE)
            w, h = img.size
        left = max(0, int(round((w - self.crop_w) * 0.5)))
        top  = max(0, int(round((h - self.crop_h) * 0.5)))
        return img.crop((left, top, left + self.crop_w, top + self.crop_h))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = self._resize_shortest(img)
        img = self._center_crop(img)
        arr = (np.asarray(img, dtype=np.float32) / 255.0)  # HWC
        if arr.ndim != 3 or arr.shape[2] != 3:
            img = img.convert("RGB")
            arr = (np.asarray(img, dtype=np.float32) / 255.0)
        # normalize
        arr = (arr - self.mean) / self.std
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW float32
        return t

# --- dataset (picklable) ---
class NudeNetSigLIPDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, Any]], preprocess: SigLIPPreprocess):
        self.rows = rows
        self.preprocess = preprocess
        self.max_pixels = int(90_000_000)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        name = _name_of(ex)
        p = ex.get("image", None)
        try:
            if p is None:
                return {"ok": False, "why": "BAD", "msg": "image path is None", "name": name}
            with Image.open(p) as img:
                img = img.convert("RGB")
                w, h = img.size
                if (w * h) > self.max_pixels:
                    return {"ok": False, "why": "BIG", "msg": f"{w}x{h}", "name": name}
                t = self.preprocess(img)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            return {"ok": False, "why": "BAD", "msg": repr(e), "name": name}
        except Exception as e:
            return {"ok": False, "why": "OTHER", "msg": repr(e), "name": name}
        y = 1 if str(ex.get("safety_label","")).lower() == "unsafe" else 0
        return {"ok": True, "x": t, "y": int(y), "name": name}

# --- collate (top-level, picklable) ---
class CollateSiglip:
    def __init__(self, log_path: str, crop_h: int, crop_w: int):
        self.log_path = log_path
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, batch: List[Dict[str, Any]]):
        xs, ys, skipped = [], [], []
        for it in batch:
            if not it.get("ok", False):
                skipped.append(f"{it.get('why','SKIP')}\t{it.get('msg','')}\t{it.get('name','<unknown>')}\n")
                continue
            xs.append(it["x"]); ys.append(it["y"])
        if skipped:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.writelines(skipped)
        if xs:
            x = torch.stack(xs, 0)
            y = torch.tensor(ys, dtype=torch.int64)
        else:
            x = torch.empty((0, 3, self.crop_h, self.crop_w), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.int64)
        return x, y, len(batch)

@torch.inference_mode()
def dump_split(model, device, processor, split: str, sexy_policy: str, bs: int = 96):
    out_prefix = "outputs/siglip_nudenet"
    rows = load_nudenet(split, sexy_policy)
    total = len(rows)
    tag = _slug(split)
    log_path = f"{out_prefix}_{tag}_skipped.txt"

    preprocess = SigLIPPreprocess(processor)
    ds = NudeNetSigLIPDataset(rows, preprocess)
    num_workers = max(1, (os.cpu_count() or 2) // 4)
    collate = CollateSiglip(log_path, preprocess.crop_h, preprocess.crop_w)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, prefetch_factor=1,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, drop_last=False,
        collate_fn=collate,
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    with open(log_path, "w", encoding="utf-8") as f: f.write("")

    pb = ProgressBar(total, desc=f"[{split}] embed", min_interval=0.5)
    xs_parts, ys_parts = [], []

    for x_cpu, y_cpu, n_total in dl:
        pb.update(n_total)
        if x_cpu.shape[0] == 0: continue
        try:
            x = x_cpu.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                img_f = model.get_image_features(pixel_values=x)
            feats = img_f.detach().to("cpu").numpy().astype("float16")
            xs_parts.append(feats); ys_parts.append(y_cpu.numpy().astype("int8"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            B, step = x_cpu.shape[0], max(8, x_cpu.shape[0]//2)
            for i in range(0, B, step):
                xb = x_cpu[i:i+step].pin_memory().to(device, non_blocking=True, memory_format=torch.channels_last)
                yb = y_cpu[i:i+step]
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fb = model.get_image_features(pixel_values=xb)
                xs_parts.append(fb.detach().to("cpu").numpy().astype("float16"))
                ys_parts.append(yb.numpy().astype("int8"))
        finally:
            torch.cuda.empty_cache()

    pb.close()

    if xs_parts:
        X = np.concatenate(xs_parts, 0).astype("float16")
        y = np.concatenate(ys_parts, 0).astype("int8")
    else:
        out_dim = int(getattr(model.config, "projection_dim", 768))
        X = np.zeros((0, out_dim), dtype="float16")
        y = np.zeros((0,), dtype="int8")

    np.save(f"{out_prefix}_{tag}_X.npy", X)
    np.save(f"{out_prefix}_{tag}_y.npy", y)
    print(f"\n[{split}] DONE kept={y.size} of total={total} bs={bs}")
    print(f"skipped log -> {log_path}")
    if y.size: print(f"{split} X: {X.shape}, y-pos-rate: {float(y.mean()):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=False)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    model_id = cfg.get("model_id", "google/siglip-so400m-patch14-384")
    sexy_policy = str(cfg.get("sexy_policy"))
    train_split = str(cfg.get("train_split"))
    val_split   = str(cfg.get("val_split"))
    test_split  = str(cfg.get("test_split"))
    bs = int(cfg.get("batch_size", 96))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(model_id, dtype=(torch.float16 if device=="cuda" else torch.float32))
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_id)

    for split in [train_split, val_split, test_split]:
        dump_split(model, device, processor, split, sexy_policy, bs)

if __name__ == "__main__":
    main()
