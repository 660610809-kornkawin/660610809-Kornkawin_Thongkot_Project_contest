import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from dataset_pair import PairDataset
from model import FoodPreferenceNet


# ──────────────────────────────────────────
# CONFIG — แก้ตรงนี้ให้ตรงกับเครื่องคุณ
# ──────────────────────────────────────────
CSV_PATH   = r"C:\Users\User\Desktop\Food\data\data_from_questionaire.csv"
IMAGE_DIR  = r"C:\Users\User\Desktop\Food\Questionair Images"
SAVE_PATH  = r"C:\Users\User\Desktop\Food\food_model_best.pth"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 3e-4
VAL_RATIO  = 0.15
# ──────────────────────────────────────────


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    desc = "  Train" if is_train else "    Val"
    bar = tqdm(loader, desc=desc, leave=False,
               bar_format="{l_bar}{bar:30}{r_bar}")

    with torch.set_grad_enabled(is_train):
        for img1, img2, label in bar:
            img1  = img1.to(device)
            img2  = img2.to(device)
            label = label.to(device)

            logit = model(img1, img2).squeeze(1)
            loss  = criterion(logit, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)

            pred       = (torch.sigmoid(logit) > 0.5).long()
            label_hard = (label > 0.5).long()
            correct   += (pred == label_hard).sum().item()
            total     += len(label)

            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.1f}%"
            )

    return total_loss / total, 100 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── Transforms ──
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Dataset ──
    full_dataset = PairDataset(CSV_PATH, IMAGE_DIR, transform=None, use_soft_label=True)
    total        = len(full_dataset)
    val_size     = int(total * VAL_RATIO)
    train_size   = total - val_size

    train_indices, val_indices = random_split(
        range(total), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = PairDataset(CSV_PATH, IMAGE_DIR, transform=train_transform, use_soft_label=True)
    val_dataset   = PairDataset(CSV_PATH, IMAGE_DIR, transform=val_transform,   use_soft_label=True)

    train_set = Subset(train_dataset, train_indices.indices)
    val_set   = Subset(val_dataset,   val_indices.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_set)} pairs | Val: {len(val_set)} pairs\n")

    # ── Model ──
    model     = FoodPreferenceNet().to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LR * 0.1},
        {"params": model.fc.parameters(),      "lr": LR},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training Loop ──
    best_val_acc     = 0
    patience         = 5
    patience_counter = 0

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs",
                     bar_format="{l_bar}{bar:20}{r_bar}")

    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        scheduler.step()

        epoch_bar.set_postfix(
            train_acc=f"{train_acc:.1f}%",
            val_acc=f"{val_acc:.1f}%",
            loss=f"{train_loss:.4f}"
        )

        tqdm.write(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Val: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            tqdm.write(f"  ✅ Best model saved! ({val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"  ⛔ Early stopping at epoch {epoch}")
                break

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
