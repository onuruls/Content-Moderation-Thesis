# src/eval/ft/dump_embeddings.py
import argparse, yaml, os, re, warnings, numpy as np, torch
import open_clip
from PIL import Image, ImageFile, UnidentifiedImageError, Image as PILImage
from src.data.nudenet import load_nudenet, iter_images as iter_images_nudenet
from src.eval.progress import ProgressBar

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def _name_of(ex) -> str:
    for k in ["path","filepath","file","image_path","source","id","uid","name"]:
        if isinstance(ex, dict) and k in ex and ex[k] is not None:
            return str(ex[k])
    return "<unknown>"

def safe_iter(gen, log_f):
    """iter_images_nudenet sırasında oluşan hataları yakala ve devam et."""
    it = iter(gen)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            msg = repr(e)
            # PIL UnidentifiedImageError mesajında path var → log’a yaz
            print(f"[skip iter] {msg}")
            log_f.write(f"ITER\t{msg}\n")
            continue

@torch.inference_mode()
def dump_split(model, preprocess, device, split, sexy_policy, out_prefix,
               bs=80, max_pixels=90_000_000):
    ds = load_nudenet(split, sexy_policy)
    it_raw = iter_images_nudenet(ds)

    total = len(ds) if hasattr(ds, "__len__") else 1
    pb = ProgressBar(total, desc=f"[{split}] embed", min_interval=0.5)

    # Hız/VRAM ayarları
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    tag = _slug(split)
    log_path = f"{out_prefix}_{tag}_skipped.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    xs, ys, buf = [], [], []

    def flush_batch():
        nonlocal xs, ys, buf
        if not buf: return
        x_cpu = torch.stack([t for (t,_) in buf], 0)  # [B,C,H,W]
        y_cpu = torch.tensor([y for (_,y) in buf], dtype=torch.float32)
        try:
            x = x_cpu.pin_memory().to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                f = model.encode_image(x)               # [B,D], fp16 hesap
            f = f.detach().to("cpu").numpy().astype("float16")
            xs.append(f); ys.append(y_cpu.numpy().astype("int8"))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = max(8, len(buf)//2)
            for i in range(0, len(buf), half):
                sub = buf[i:i+half]
                x_cpu2 = torch.stack([t for (t,_) in sub], 0)
                y_cpu2 = torch.tensor([y for (_,y) in sub], dtype=torch.float32)
                xb = x_cpu2.pin_memory().to(device, non_blocking=True, memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    fb = model.encode_image(xb)
                fb = fb.detach().to("cpu").numpy().astype("float16")
                xs.append(fb); ys.append(y_cpu2.numpy().astype("int8"))
        finally:
            buf = []
            torch.cuda.empty_cache()

    i_seen = 0
    # Jeneratörü güvenli sarmalayıcıyla gez
    for ex in safe_iter(it_raw, log_f):
        i_seen += 1
        pb.update(1)
        name = _name_of(ex)
        img = ex.get("image", None)

        try:
            if img is None:
                raise OSError("image is None")
            if isinstance(img, (str, bytes, os.PathLike)):
                with PILImage.open(img) as _im:
                    img = _im.convert("RGB")
            else:
                w, h = img.size
                if (w*h) > max_pixels:
                    log_f.write(f"BIG\t{w}x{h}\t{name}\n")
                    continue
                img = img.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as e:
            log_f.write(f"BAD\t{repr(e)}\t{name}\n")
            continue
        except Exception as e:
            log_f.write(f"OTHER\t{repr(e)}\t{name}\n")
            continue

        try:
            t = preprocess(img)  # CPU tensor
        except Exception as e:
            log_f.write(f"PREPROC\t{repr(e)}\t{name}\n")
            continue

        buf.append((t, int(ex["label"])))
        if len(buf) >= bs:
            flush_batch()

    flush_batch()
    log_f.close()
    pb.close()

    if xs:
        X = np.concatenate(xs, 0).astype("float16")
        y = np.concatenate(ys, 0).astype("int8")
    else:
        out_dim = getattr(model.visual, "output_dim", 768)
        X = np.zeros((0, out_dim), dtype="float16")
        y = np.zeros((0,), dtype="int8")

    np.save(f"{out_prefix}_{tag}_X.npy", X)
    np.save(f"{out_prefix}_{tag}_y.npy", y)
    print(f"\n[{split}] DONE kept={len(y)} bs={bs}")
    print(f"skipped log -> {log_path}")
    print(split, X.shape, (y.mean() if len(y) else 'n/a'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out_prefix", default="outputs/clip_vitl14_nudenet")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--max_pixels", type=int, default=90_000_000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg,"r", encoding="utf-8"))
    name, pretrained = cfg["model_id"], cfg.get("pretrained","openai")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    model.eval().requires_grad_(False)

    train_split = str(cfg.get("train_split"))
    val_split   = str(cfg.get("val_split"))
    test_split  = str(cfg.get("test_split"))

    cfg_bs = int(cfg.get("batch_size", 80))
    bs = int(args.bs) if args.bs is not None else cfg_bs
    sexy_policy = str(cfg.get("sexy_policy"))

    for split in [train_split, val_split, test_split]:
        dump_split(model, preprocess, device, split, sexy_policy, args.out_prefix,
                   bs=bs, max_pixels=args.max_pixels)

if __name__ == "__main__":
    main()
