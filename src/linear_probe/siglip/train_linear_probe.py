import argparse, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score

prefix = "outputs/siglip_nudenet"
l2norm = True

# ---------- Data ----------
class NpDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, memmap: bool):
        self.X = X
        self.y = y.astype(np.float32)
        self.memmap = memmap  # if True, X may be memory-mapped (float16 on disk)

    def __len__(self): return int(self.X.shape[0])

    def __getitem__(self, i):
        # Load row; cast to float32 for numeric stability
        xi = self.X[i]  # may be float16 (memmap) or float32
        if isinstance(xi, np.ndarray):
            x = torch.from_numpy(xi.astype(np.float32, copy=False))
        else:
            # Fallback in rare slicing cases
            x = torch.tensor(xi, dtype=torch.float32)

        if l2norm:
            n = torch.linalg.norm(x) + 1e-12
            x = x / n
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y

def load_xy(tag: str, memmap: bool):
    X = np.load(f"{prefix}_{tag}_X.npy", mmap_mode="r" if memmap else None)
    y = np.load(f"{prefix}_{tag}_y.npy")
    return X, y

# ---------- Model ----------
class LinearHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w + self.b).squeeze(-1)

# ---------- Train / Eval ----------
def train_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logit = model(xb)
        loss = crit(logit, yb)
        loss.backward()
        opt.step()
        total += float(loss.item()) * xb.size(0)
    return total / max(1, len(loader.dataset))

@torch.inference_mode()
def eval_epoch(model, loader, device):
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logit = model(xb)
        p = torch.sigmoid(logit).cpu().numpy()
        ps.append(p); ys.append(yb.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    ap = float(average_precision_score(y, p))
    try: ra = float(roc_auc_score(y, p))
    except Exception: ra = 0.0
    return ap, ra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tag", default="train_100", help="suffix between prefix_ and _X.npy for train")
    ap.add_argument("--val_tag",   default="validation_50", help="suffix for validation")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--memmap", action="store_true", help="load arrays with mmap_mode='r' to save RAM")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    Xtr, ytr = load_xy(args.train_tag, args.memmap)
    Xva, yva = load_xy(args.val_tag,   args.memmap)

    # Dataset / loaders
    tr_ds = NpDataset(Xtr, ytr, args.memmap)
    va_ds = NpDataset(Xva, yva, args.memmap)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = int(Xtr.shape[1])

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = LinearHead(D).to(device)

    # Imbalance handling
    n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
    pos_w = n_neg / max(1.0, n_pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ap, best_state = -1.0, None
    patience, bad = 2, 0

    try:
        for ep in range(1, args.epochs + 1):
            tl = train_epoch(model, tr_loader, crit, opt, device)
            ap, ra = eval_epoch(model, va_loader, device)
            print(f"epoch {ep:02d} | train_loss {tl:.5f} | val PR-AUC {ap:.4f} ROC-AUC {ra:.4f}")

            if ap > best_ap + 1e-6:
                best_ap = ap; bad = 0
                best_state = {"w": model.w.detach().cpu().numpy(), "b": model.b.detach().cpu().numpy()}
                np.savez(f"{prefix}_linear_head_best.npz", **best_state)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stop (patience={patience}).")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user — saving best so far...")
    finally:
        if best_state is not None:
            np.savez(f"{prefix}_linear_head_best.npz", **best_state)
            print("saved:", f"{prefix}_linear_head_best.npz")
        else:
            print("No best state to save.")

if __name__ == "__main__":
    main()
