"""
Custom CNN Character Recognizer for Handwritten Form Fields.
Optimized for: Numbers (0-9), Dates (0-9, '/', '-', '.'), and Alphanumeric Short Codes.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_DATE_NUMERIC = "0123456789/-."
VOCAB_ALPHANUMERIC = "0123456789/-.ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class FieldCharacterCNN(nn.Module):
    """
    High-accuracy, lightweight Convolutional Neural Network for handwritten form character recognition.
    Processes normalized 32x32 grayscale glyphs.
    """

    def __init__(self, num_classes: int = len(VOCAB_ALPHANUMERIC)):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 32x32 -> 16x16
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 16x16 -> 8x8
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 8x8 -> 4x4
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop_conv = nn.Dropout2d(0.20)

        # Dense Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.40)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(x)
        x = self.drop_conv(x)

        # Dense
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc(x)
        logits = self.fc_out(x)
        return logits

    def predict_patch(
        self,
        patch_tensor: torch.Tensor,
        vocab: str = VOCAB_ALPHANUMERIC,
        allowed_vocab: Optional[str] = None,
        device: str = "cuda"
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Runs inference on a single (1, 1, 32, 32) tensor.
        Args:
            patch_tensor: normalized 32x32 tensor
            vocab: full model vocabulary string
            allowed_vocab: optional string restricting classes (e.g. for Dates or PIN codes)
            device: 'cuda' or 'cpu'
        Returns:
            - predicted_char: Best class string
            - confidence: Softmax probability [0, 1]
            - alternatives: Top-3 alternative candidates with probabilities
        """
        self.eval()
        with torch.no_grad():
            x = patch_tensor.to(device)
            logits = self(x)[0]

            if allowed_vocab is not None and len(allowed_vocab) > 0:
                allowed_indices = [vocab.index(c) for c in allowed_vocab if c in vocab]
                if allowed_indices:
                    mask = torch.full_like(logits, -1e9)
                    mask[allowed_indices] = logits[allowed_indices]
                    logits = mask

            probs = torch.softmax(logits, dim=-1)
            k_val = min(4, len(allowed_vocab) if allowed_vocab else len(vocab))
            topk = torch.topk(probs, k=k_val)

            pred_idx = topk.indices[0].item()
            pred_char = vocab[pred_idx] if pred_idx < len(vocab) else "?"
            confidence = float(topk.values[0].item())

            alts = []
            for i in range(1, len(topk.indices)):
                idx = topk.indices[i].item()
                c = vocab[idx] if idx < len(vocab) else "?"
                p = float(topk.values[i].item())
                alts.append((c, round(p, 4)))

        return pred_char, confidence, alts
