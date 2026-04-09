import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from util import compute_eer

"""
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_channels, attention_channels=128):
        super(AttentiveStatsPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        w = self.attention(x)
        w = F.softmax(w, dim=2)

        mean = torch.sum(w * x, dim=2)
        var = torch.sum(w * (x - mean.unsqueeze(2)) ** 2, dim=2)
        std = torch.sqrt(var.clamp(min=1e-9))

        return torch.cat([mean, std], dim=1)

class myTCN(nn.Module):
    def __init__(self):
        super(myTCN, self).__init__()

        # 768 -> 384
        self.input_proj = nn.Conv1d(768, 384, kernel_size=1)

        # 2-layer TCN: 384 -> 192 -> 96
        self.tcn1 = nn.Conv1d(384, 192, kernel_size=3, dilation=1, padding=1)
        self.tcn2 = nn.Conv1d(192, 96, kernel_size=3, dilation=2, padding=2)

        self.bn1 = nn.BatchNorm1d(192)
        self.bn2 = nn.BatchNorm1d(96)

        self.dropout = nn.Dropout(0.25)

        # last TCN output = 96 channels
        self.pool = AttentiveStatsPool(in_channels=96, attention_channels=48)

        # pool output = 2 * 96 = 192
        self.fc1 = nn.Linear(192, 64)
        #self.fc2 = nn.Linear(96, 48)
        self.out = nn.Linear(64, 1)

    def forward(self, X: torch.Tensor):
        X = X.transpose(1, 2)            # [B, 768, T]
        X = self.input_proj(X)           # [B, 384, T]

        X = self.dropout(F.gelu(self.bn1(self.tcn1(X))))  # [B, 192, T]
        X = self.dropout(F.gelu(self.bn2(self.tcn2(X))))  # [B, 96, T]

        X = self.pool(X)                 # [B, 192]

        X = self.dropout(F.gelu(self.fc1(X)))  # [B, 96]
        #X = self.dropout(F.gelu(self.fc2(X)))  # [B, 48]
        X = self.out(X)                        # [B, 1]

        return X
"""

class AttentiveStatsPoolSmall(nn.Module):
    def __init__(self, in_channels=768, attention_channels=128, output_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1)
        )

        self.proj = nn.Linear(in_channels * 2, output_dim)

        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, 768, T]

        w = self.attention(x)
        w = F.softmax(w, dim=2)

        mean = torch.sum(w * x, dim=2)
        var = torch.sum(w * (x - mean.unsqueeze(2)) ** 2, dim=2)
        std = torch.sqrt(var.clamp(min=1e-9))

        pooled = torch.cat([mean, std], dim=1)  # [B, 1536]

        pooled = self.proj(pooled)   # [B, 128]
        #pooled = self.act(pooled)    # 🔥 activation here

        return pooled


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = AttentiveStatsPoolSmall(768, 384, 768)
        self.l1 = nn.Linear(768, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(x)
        x = F.gelu((self.l1(x)))
        x = self.out(x)   # no activation (BCEWithLogitsLoss expects raw logits)
        return x

def train_model(model:SmallModel, train_loader, val_loader, epochs=10, device="cuda"):
    model.to(device)
    """
    for params in model.parameters():
        params.requires_grad = False
    
    for module in [ model.out, model.fc1]:
        for param in module.parameters():
            param.requires_grad = True
    """
    criterion = nn.BCEWithLogitsLoss()
    smoothing = 0.15

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=9e-5,
        weight_decay=4e-4
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # =========================
        # TRAIN
        # =========================
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc="Training")

        for batch in loop:
            if len(batch) == 2:
                X, y = batch
            else:
                X, y, _ = batch

            X = X.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            logits = model(X).squeeze(1)

            # weight spoof class more to reduce false bonafide
            pos_weight = torch.tensor([2.5], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            smoothed_y = torch.where(y == 1.0, 1.0 - smoothing, smoothing)

            # 3. Calculate loss using the smoothed labels and training criterion
            loss = criterion(logits, smoothed_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss = 0.0

        # -------- chunk-level tracking --------
        chunk_correct = 0
        chunk_total = 0
        chunk_false_bonafide = 0  # y=1, pred=0
        chunk_false_spoof = 0     # y=0, pred=1
        chunk_scores = []
        chunk_labels = []

        # -------- file-level tracking --------
        file_probs = {}
        file_logits = {}
        file_labels = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if len(batch) != 3:
                    raise ValueError(
                        "val_loader must return (X, y, filenames) for validation."
                    )

                X, y, filenames = batch
                X = X.to(device)
                y = y.float().to(device)

                logits = model(X).squeeze(1)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).detach().cpu()
                preds = (probs > 0.5)

                y_cpu = y.detach().cpu()
                logits_cpu = logits.detach().cpu()

                # =========================
                # CHUNK-LEVEL STATS
                # =========================
                chunk_correct += (preds == y_cpu.bool()).sum().item()
                chunk_total += y_cpu.size(0)

                chunk_false_bonafide += ((y_cpu == 1) & (preds == 0)).sum().item()
                chunk_false_spoof += ((y_cpu == 0) & (preds == 1)).sum().item()

                chunk_scores.extend(probs.numpy().tolist())
                chunk_labels.extend(y_cpu.numpy().tolist())

                # =========================
                # FILE-LEVEL STATS
                # =========================
                for i, fname in enumerate(filenames):
                    if fname not in file_probs:
                        file_probs[fname] = []
                        file_logits[fname] = []
                        file_labels[fname] = int(y_cpu[i].item())

                    file_probs[fname].append(float(probs[i].item()))
                    file_logits[fname].append(float(logits_cpu[i].item()))

        val_loss /= len(val_loader)

        # =========================
        # CHUNK-LEVEL METRICS
        # =========================
        chunk_acc = chunk_correct / chunk_total
        chunk_false_bonafide_rate = chunk_false_bonafide / chunk_total
        chunk_false_spoof_rate = chunk_false_spoof / chunk_total
        chunk_eer = compute_eer(
            np.array(chunk_labels),
            np.array(chunk_scores)
        )

        # =========================
        # FILE-LEVEL METRICS
        # =========================
        file_correct = 0
        file_total = 0
        file_false_bonafide = 0
        file_false_spoof = 0

        final_scores = []
        final_labels = []

        for fname in file_probs:
            logits_list = file_logits[fname]

            logits_list = file_logits[fname]

            file_logit = float(np.median(logits_list))
            file_prob = torch.sigmoid(torch.tensor(file_logit)).item()
            pred = 1 if file_prob > 0.5 else 0

            label = file_labels[fname]
            pred = 1 if file_prob > 0.5 else 0

            if pred == label:
                file_correct += 1

            if label == 1 and pred == 0:
                file_false_bonafide += 1

            if label == 0 and pred == 1:
                file_false_spoof += 1

            file_total += 1
            final_scores.append(file_prob)
            final_labels.append(label)

        file_acc = file_correct / file_total
        file_false_bonafide_rate = file_false_bonafide / file_total
        file_false_spoof_rate = file_false_spoof / file_total
        file_eer = compute_eer(
            np.array(final_labels),
            np.array(final_scores)
        )

        # =========================
        # PRINT
        # =========================
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        print("\n--- Chunk-level / all voice batches ---")
        print(f"Chunk Acc:            {chunk_acc:.4f}")
        print(f"Chunk False Bonafide: {chunk_false_bonafide_rate * 100:.2f}%")
        print(f"Chunk False Spoof:    {chunk_false_spoof_rate * 100:.2f}%")
        print(f"Chunk EER:            {chunk_eer:.2f}%")

        print("\n--- File-level ---")
        print(f"File Acc:             {file_acc:.4f}")
        print(f"File False Bonafide:  {file_false_bonafide_rate * 100:.2f}%")
        print(f"File False Spoof:     {file_false_spoof_rate * 100:.2f}%")
        print(f"File EER:             {file_eer:.2f}%")

