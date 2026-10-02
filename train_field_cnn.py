"""
Training Script for Custom FieldCharacterCNN.
Trains a lightweight multi-layer CNN on defined character vocabulary (0-9, '/', '-', '.', A-Z).
Runs on NVIDIA GPU in ~40 seconds and saves models/field_cnn.pth.
"""

from __future__ import annotations

import os
import sys
import time
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Setup paths
repo_root = r"C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon"
sys.path.insert(0, repo_root)
sys.path.insert(0, r"C:\Users\SRIRAM\.gemini\antigravity-ide\brain\4470012d-eb77-4c69-b7a1-555f936d8bda\scratch")

try:
    from src.field_reader.model import FieldCharacterCNN, VOCAB_ALPHANUMERIC
    from src.field_reader.dataset import CharacterDataset
except ImportError:
    from model import FieldCharacterCNN, VOCAB_ALPHANUMERIC
    from dataset import CharacterDataset

def train_cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training FieldCharacterCNN on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Character Vocabulary ({len(VOCAB_ALPHANUMERIC)} classes): '{VOCAB_ALPHANUMERIC}'")

    # 1. Dataset Preparation
    print("Generating synthetic & augmented character dataset (500 samples per class)...")
    dataset = CharacterDataset(vocab=VOCAB_ALPHANUMERIC, samples_per_class=500)
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    print(f"Dataset ready: {len(train_ds)} training glyphs, {len(val_ds)} validation glyphs.")

    # 2. Model Initialization
    model = FieldCharacterCNN(num_classes=len(VOCAB_ALPHANUMERIC)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    # 3. Training Loop
    epochs = 12
    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    out_dir = os.path.join(repo_root, "models")
    os.makedirs(out_dir, exist_ok=True)
    model_save_path = os.path.join(out_dir, "field_cnn.pth")

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = (correct / total) * 100.0
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:6.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": VOCAB_ALPHANUMERIC,
                "best_acc": best_acc,
                "num_classes": len(VOCAB_ALPHANUMERIC),
            }, model_save_path)

    elapsed = time.time() - t_start
    print(f"\n✅ Training Completed in {elapsed:.1f}s!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {model_save_path}")

    # Save metrics JSON
    with open(os.path.join(out_dir, "field_cnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train_cnn()
