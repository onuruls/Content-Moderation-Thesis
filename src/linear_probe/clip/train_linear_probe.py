# src/eval/ft/train_linear_probe.py
import argparse, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW

class NpDataset(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]).float(), torch.tensor(self.y[i],dtype=torch.float32)

class LinearHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # CLIP feature -> 1 logit (bias dahil)
        self.w = nn.Parameter(torch.zeros(dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        # x: [B, D]
        return (x @ self.w + self.b).squeeze(-1)  # [B]

def train_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logit = model(xb)
        loss = crit(logit, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total/len(loader.dataset)

@torch.inference_mode()
def eval_epoch(model, loader, device):
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb)
        p = torch.sigmoid(logit).cpu().numpy()
        ps.append(p); ys.append(yb.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    ap = float(average_precision_score(y, p))
    try: ra = float(roc_auc_score(y, p))
    except: ra = 0.0
    return ap, ra, p, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--l2norm", action="store_true", help="row-wise L2 normalize features")
    args = ap.parse_args()

    # Dosyalar: _train_X/y.npy ve _validation_X/y.npy
    Xtr = np.load(f"{args.prefix}_train_X.npy").astype("float32")
    ytr = np.load(f"{args.prefix}_train_y.npy").astype("int64")
    Xva = np.load(f"{args.prefix}_val_X.npy").astype("float32")
    yva = np.load(f"{args.prefix}_val_y.npy").astype("int64")

    # (Opsiyonel) satır bazında L2 normalizasyon
    if args.l2norm:
        def l2n(X):
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / n
        Xtr, Xva = l2n(Xtr), l2n(Xva)

    D = Xtr.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LinearHead(D).to(device)
    tr_loader = torch.utils.data.DataLoader(NpDataset(Xtr,ytr), batch_size=args.bs, shuffle=True)
    va_loader = torch.utils.data.DataLoader(NpDataset(Xva,yva), batch_size=args.bs, shuffle=False)

    # Imbalance için pozitif sınıf ağırlığı
    pos_w = (len(ytr)-int(ytr.sum()))/max(1,int(ytr.sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ap, best_state = -1.0, None
    patience, bad = 2, 0  # 2 epoch iyileşme yoksa dur

    try:
        for ep in range(1, args.epochs + 1):
            tl = train_epoch(model, tr_loader, crit, opt, device)
            ap, ra, p, y = eval_epoch(model, va_loader, device)
            print(f"epoch {ep:02d} | train_loss {tl:.4f} | val PR-AUC {ap:.4f} ROC-AUC {ra:.4f}")

            if ap > best_ap + 1e-5:  # küçük bir marj
                best_ap = ap;
                bad = 0
                best_state = {"w": model.w.detach().cpu().numpy(),
                              "b": model.b.detach().cpu().numpy()}
                # HER iyileşmede kaydet (güvenli)
                np.savez(f"{args.prefix}_linear_head_best.npz", **best_state)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stop (patience={patience}).")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user — saving best so far...")
    finally:
        if best_state is not None:
            np.savez(f"{args.prefix}_linear_head_best.npz", **best_state)
            print("saved:", f"{args.prefix}_linear_head_best.npz")
        else:
            print("No best state to save.")


if __name__ == "__main__":
    main()
