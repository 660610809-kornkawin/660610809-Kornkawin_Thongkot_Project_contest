import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FoodPreferenceNet
from dataset_folder_pair import FolderPairDataset


# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
IMAGE_DIR  = r"C:\Users\User\Desktop\Food\data\Pizza"
MODEL_PATH = "food_model_best.pth"
OUTPUT     = "results/Pizza_test.xlsx"
# ──────────────────────────────────────────


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

dataset = FolderPairDataset(IMAGE_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

model = FoodPreferenceNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model.eval()

results = []

with torch.no_grad():
    bar = tqdm(loader, desc="Testing", bar_format="{l_bar}{bar:30}{r_bar}")
    for img1, img2, name1, name2 in bar:
        logit = model(img1, img2)
        prob  = torch.sigmoid(logit).item()
        winner = 1 if prob > 0.5 else 2

        results.append({
            "image1": name1[0],
            "image2": name2[0],
            "prob_img1_wins": round(prob, 4),
            "winner": winner,
        })

        bar.set_postfix(pair=f"{name1[0]} vs {name2[0]}", winner=f"img{winner}")

df = pd.DataFrame(results)
df.to_excel(OUTPUT, index=False)
print(f"\n✅ Saved {len(df)} results → {OUTPUT}")
