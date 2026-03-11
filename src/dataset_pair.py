import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, use_soft_label=True):
        """
        Args:
            csv_path     : path ของไฟล์ data_from_questionaire.csv
            image_dir    : folder ที่เก็บรูปทั้งหมด (ไม่ต้องแยก subfolder)
            transform    : torchvision transforms
            use_soft_label: True  = label เป็น vote ratio เช่น 81/(81+48) = 0.628
                           False = label เป็น 1.0 หรือ 0.0 (hard label)
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.use_soft_label = use_soft_label

        # สลับคู่ (data augmentation) — img1↔img2, label กลับด้าน
        original = self.df.copy()
        flipped = self.df.copy()
        flipped["Image 1"] = original["Image 2"]
        flipped["Image 2"] = original["Image 1"]
        flipped["Num Vote 1"] = original["Num Vote 2"]
        flipped["Num Vote 2"] = original["Num Vote 1"]
        flipped["Winner"] = original["Winner"].apply(lambda w: 2 if w == 1 else 1)

        self.df = pd.concat([original, flipped], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1_path = os.path.join(self.image_dir, row["Image 1"])
        img2_path = os.path.join(self.image_dir, row["Image 2"])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.use_soft_label:
            v1 = row["Num Vote 1"]
            v2 = row["Num Vote 2"]
            label = v1 / (v1 + v2)  # float 0.0–1.0 (>0.5 = img1 ชนะ)
        else:
            label = 1.0 if row["Winner"] == 1 else 0.0

        return img1, img2, float(label)
