import torch
import torch.nn as nn
import os
import time
import json
import numpy as np
from tqdm import tqdm
from cmt.train.losses import compute_pos_weight, AsymmetricLoss
from cmt.eval.calibrate import choose_threshold_by_f1
from cmt.utils.io_utils import ensure_dir

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.fc(x)

class Trainer:
    def __init__(self, cfg, backbone, preprocess, train_loader, val_loader, class_names):
        self.cfg = cfg
        self.backbone = backbone
        self.preprocess = preprocess  # Not used inside loop if loader yields tensors, but kept for context
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_dir = os.path.join("results", "train", cfg.get("run_name", f"run_{int(time.time())}"))
        ensure_dir(self.run_dir)
        
        # Setup Head
        if hasattr(backbone, "visual") and hasattr(backbone.visual, "output_dim"):
            self.feat_dim = backbone.visual.output_dim
        elif hasattr(backbone, "num_features"):
             self.feat_dim = backbone.num_features
        else:
             self.feat_dim = cfg.get("feature_dim", 768)
             
        self.model = LinearHead(self.feat_dim, len(class_names)).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=float(cfg.get("lr", 1e-3)), 
            weight_decay=float(cfg.get("weight_decay", 1e-2))
        )
        
        # Loss
        loss_type = cfg.get("loss", "bce")
        if loss_type == "bce":
            if hasattr(train_loader.dataset, "df"):
                pw = compute_pos_weight(train_loader.dataset.df, class_names)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "asl":
            self.criterion = AsymmetricLoss(
                gamma_neg=cfg.get("asl_gamma_neg", 4),
                gamma_pos=cfg.get("asl_gamma_pos", 1),
                clip=cfg.get("asl_clip", 0.05)
            )
            
    def encode(self, images):
        # images: Tensor [B, C, H, W]
        with torch.no_grad():
            if self.cfg.get("use_amp", True) and self.device == "cuda":
                 with torch.amp.autocast("cuda"):
                     features = self.backbone.encode_image(images)
            else:
                 features = self.backbone.encode_image(images)
            
            # L2 Norm?
            if self.cfg.get("l2_norm", True):
                features = features / features.norm(dim=-1, keepdim=True)
                
            return features.float()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch} Train")
        for imgs, labels in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device)
            
            features = self.encode(imgs)
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(imgs)
            count += len(imgs)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / count
        
    def validate(self, epoch):
        self.model.eval()
        ys, ss = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc=f"Ep {epoch} Val"):
                imgs = imgs.to(self.device, non_blocking=True)
                features = self.encode(imgs)
                logits = self.model(features)
                probs = torch.sigmoid(logits)
                
                ys.append(labels.cpu().numpy())
                ss.append(probs.cpu().numpy())
                
        Y = np.vstack(ys)
        S = np.vstack(ss)
        
        # Macro F1 at best thresholds
        # Per class thresholding
        f1s = []
        thresholds = {}
        for i, name in enumerate(self.class_names):
            y_c = Y[:, i]
            s_c = S[:, i]
            t, f1 = choose_threshold_by_f1(y_c, s_c)
            f1s.append(f1)
            thresholds[name] = t
            
        macro_f1 = np.mean(f1s)
        return macro_f1, thresholds

    def fit(self):
        epochs = self.cfg.get("epochs", 10)
        best_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            val_f1, thrs = self.validate(epoch)
            
            print(f"Epoch {epoch}: Loss={loss:.4f}, Val Macro F1={val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save
                self.save(f"best_model_ep{epoch}.pt", thrs)
                self.save("best_model.pt", thrs) # overwrite best
                
    def save(self, filename, thresholds):
        path = os.path.join(self.run_dir, filename)
        state = {
            "state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "thresholds": thresholds,
            "cfg": self.cfg
        }
        torch.save(state, path)
        with open(path.replace(".pt", "_thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)
