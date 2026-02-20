import argparse
import yaml
import torch
import open_clip
from cmt.train.trainer import Trainer
from cmt.data.training import MultiLabelCsvDataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Backbone
    print(f"Loading backbone: {cfg['arch']} ({cfg['pretrained']})")
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        cfg["arch"], 
        pretrained=cfg["pretrained"], 
        device=device
    )
    backbone.eval()
    backbone.requires_grad_(False)
    
    # 2. Dataset
    print(f"Loading datasets from {cfg['train_csv']} and {cfg['val_csv']}...")
    train_csv = cfg["train_csv"]
    val_csv = cfg["val_csv"]
    img_root = cfg["image_root"]
    
    class_names = cfg.get("class_names")
    if not class_names:
        raise ValueError("Config must specify class_names list")
        
    ds_train = MultiLabelCsvDataset(train_csv, img_root, preprocess, class_names)
    ds_val = MultiLabelCsvDataset(val_csv, img_root, preprocess, class_names)
    
    bs = cfg.get("batch_size", 32)
    loader_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Train
    print("Starting training...")
    trainer = Trainer(cfg, backbone, preprocess, loader_train, loader_val, class_names)
    trainer.fit()

if __name__ == "__main__":
    main()
