import torch
from PIL import Image
from torchvision import transforms

from model import FoodPreferenceNet


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FoodPreferenceNet().to(device)

model.load_state_dict(torch.load("best_model.pth"))

model.eval()


def predict(img1_path, img2_path):

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():

        score = model(img1, img2)

        prob = torch.sigmoid(score).item()

    return prob


p = predict("pizza.jpg", "burger.jpg")

print("Preference score:", p)

if p > 0.5:
    print("Prefer image1")
else:
    print("Prefer image2")
