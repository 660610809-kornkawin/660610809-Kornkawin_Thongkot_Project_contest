import os
import re
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def natural_sort_key(filename):
    # แยก ตัวอักษร กับ ตัวเลข ออกจากกัน เช่น "b10_1.jpg" → ["b", 10, "_", 1, ".jpg"]
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', filename)]


class FolderPairDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".png"))
        ], key=natural_sort_key)  # ← เรียงแบบ natural sort

        self.pairs = []
        for i in range(0, len(self.images), 2):
            if i + 1 < len(self.images):
                self.pairs.append((self.images[i], self.images[i+1]))

        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]

        img1 = Image.open(os.path.join(self.image_folder, img1_name)).convert("RGB")
        img2 = Image.open(os.path.join(self.image_folder, img2_name)).convert("RGB")

        return (
            self.transform(img1),
            self.transform(img2),
            img1_name,
            img2_name,
        )
