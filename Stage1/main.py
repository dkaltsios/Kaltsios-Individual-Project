import os
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# Dataset expects a CSV with columns: image_path,is_malign (1 = malign, 0 = benign)
class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        if 'image_path' not in self.df.columns or 'is_malign' not in self.df.columns:
            raise SystemExit("CSV must contain columns: image_path,is_malign")
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / str(row['image_path'])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['is_malign'])
        return img, torch.tensor(label, dtype=torch.long)

def build_model(num_classes=2, freeze_backbone=False):
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(pretrained=True)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, out_path):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            _, pidx = preds.max(1)
            running_correct += (pidx == yb).sum().item()
            running_total += xb.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Val epoch {epoch}"):
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                _, pidx = preds.max(1)
                val_correct += (pidx == yb).sum().item()
                val_total += xb.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'val_acc': best_val_acc}, out_path)
    return history

def plot_history(history, out_file='training_history.png'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title('Accuracy %')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='skin_lesions.csv', help='CSV with image_path,is_malign')
    parser.add_argument('--img_dir', default='images', help='Directory with image files')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--out', default='best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.csv).exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not Path(args.img_dir).exists():
        raise SystemExit(f"Image directory not found: {args.img_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = SkinLesionDataset(args.csv, args.img_dir, transform=train_transform)
    n = len(dataset)
    val_size = max(1, int(0.15 * n))
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    # ensure val uses deterministic transforms
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(num_classes=2, freeze_backbone=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.out)
    plot_history(history)

    print(f"Training complete. Best model (by val acc) saved to {args.out} and history saved to training_history.png")

if __name__ == '__main__':
    main()