import torch
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
from cmt.utils.image_utils import ensure_rgb

class MultiLabelCsvDataset(Dataset):
    def __init__(self, csv_path, root_dir, preprocess, class_names):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = preprocess
        self.class_names = class_names
        
        # Verify columns
        missing = [c for c in class_names if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
            
        self.images = self.df["name"].values
        self.labels = self.df[class_names].values.astype("float32")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        
        try:
            img = Image.open(img_path)
            img = ensure_rgb(img)
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise e
            
        label = torch.tensor(self.labels[idx])
        return img_tensor, label
