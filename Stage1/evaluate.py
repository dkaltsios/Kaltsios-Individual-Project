import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        if 'image_path' not in self.df.columns:
            raise SystemExit("CSV must contain column: image_path")
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / str(row['image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['is_malign']) if 'is_malign' in self.df.columns else -1
        return img, label, str(row['image_path'])

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

def evaluate(model, loader, device):
    model.eval()
    probs_all = []
    preds_all = []
    labels_all = []
    paths_all = []
    soft = nn.Softmax(dim=1)
    with torch.no_grad():
        for xb, yb, paths in tqdm(loader, desc="Evaluating"):
            xb = xb.to(device)
            logits = model(xb)
            probs = soft(logits)[:, 1].cpu().numpy()  # probability of malign (class 1)
            preds = (probs >= 0.5).astype(int)
            probs_all.extend(probs.tolist())
            preds_all.extend(preds.tolist())
            labels_all.extend(yb.numpy().tolist() if isinstance(yb, torch.Tensor) else yb)
            paths_all.extend(paths)
    return paths_all, labels_all, preds_all, probs_all

def predict_single(model, img_path, device):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(inp)
        prob = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(prob >= 0.5)
    return pred, prob

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to saved checkpoint (best_model.pth)')
    p.add_argument('--csv', required=False, help='CSV with image_path[,is_malign] to evaluate')
    p.add_argument('--img_dir', default='images')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out', default='predictions.csv')
    p.add_argument('--single', help='Path to single image to predict')
    args = p.parse_args()

    if not args.csv and not args.single:
        p.error("either --csv or --single must be provided")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # load model early so single-image prediction works without csv/dataset
    model = build_model(num_classes=2).to(device)
    ckpt = torch.load(args.model, map_location=device)
    # ckpt may contain 'model_state' or be the state_dict directly
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)

    # single-image prediction path
    if args.single:
        single_path = Path(args.single)
        if not single_path.exists():
            raise SystemExit(f"Single image not found: {args.single}")
        pred, prob = predict_single(model, args.single, device)
        print(f"Image: {args.single}  predicted_malign={pred}  prob_malign={prob:.4f}")
        return

    # CSV / batch evaluation path
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = CsvImageDataset(args.csv, args.img_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    paths, labels, preds, probs = evaluate(model, loader, device)

    df = pd.DataFrame({'image_path': paths, 'label': labels, 'pred': preds, 'prob_malign': probs})
    df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

    if all([l in (0,1) for l in labels]):
        acc = accuracy_score(labels, preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(labels, preds, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(labels, preds))
        try:
            auc = roc_auc_score(labels, probs)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass
    else:
        print("No ground-truth labels in CSV to compute metrics.")

if __name__ == '__main__':
    main()