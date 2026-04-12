import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from util import compute_eer

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
        self.tcn1 = nn.Conv1d(768, 384, kernel_size=3, dilation=1, padding=1)
        self.tcn2 = nn.Conv1d(384, 128, kernel_size=3, dilation=2, padding=2)
        self.tcn3 = nn.Conv1d(128, 64, kernel_size=3, dilation=4, padding=4)
        
        self.bn1 = nn.BatchNorm1d(384)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.15)
        self.pool = AttentiveStatsPool(in_channels=64, attention_channels=32)

    def forward(self, X: torch.Tensor):
        X = X.transpose(1, 2)            # [B, 768, T]
        X = self.dropout(F.gelu(self.bn1(self.tcn1(X)))) 
        X = self.dropout(F.gelu(self.bn2(self.tcn2(X)))) 
        X = self.dropout(F.gelu(self.bn3(self.tcn3(X)))) 

        X = self.pool(X)                 
        return X

class AttentiveStatsPoolSmall(nn.Module):
    def __init__(self, in_channels=768, attention_channels=128, output_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1)
        )
        self.proj = nn.Linear(in_channels * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, 768, T]
        w = self.attention(x)
        w = F.softmax(w, dim=2)
        mean = torch.sum(w * x, dim=2)
        var = torch.sum(w * (x - mean.unsqueeze(2)) ** 2, dim=2)
        std = torch.sqrt(var.clamp(min=1e-9))
        pooled = torch.cat([mean, std], dim=1)  # [B, 1536]
        pooled = self.proj(pooled)              # [B, 128]
        return pooled

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = AttentiveStatsPoolSmall(768, 384, 768)
        self.l1 = nn.Linear(768, 32)

    def forward(self, x):
        x = self.pool(x)
        x = self.l1(x)  # [B, 32]
        return x

class OCSoftmax(nn.Module):
    # 🔥 UPDATED: r_real to 0.75 and r_fake to 0.15
    def __init__(self, feat_dim=128, r_real=0.75, r_fake=0.15, alpha=20.0, fake_penalty=2.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.fake_penalty = fake_penalty

        self.center = nn.Parameter(torch.randn(feat_dim, 1))
        nn.init.kaiming_uniform_(self.center, 0.25)
    
    def forward(self, x, labels):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.center, p=2, dim=0)

        scores = torch.matmul(x_norm, w_norm).squeeze(1)
        loss = torch.zeros_like(scores)

        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        if real_mask.any():
            loss[real_mask] = torch.log(1 + torch.exp(self.alpha * (self.r_real - scores[real_mask])))

        if fake_mask.any():
            loss[fake_mask] = self.fake_penalty * torch.log(1 + torch.exp(self.alpha * (scores[fake_mask] - self.r_fake)))

        return loss.mean(), scores

def train_model(model: nn.Module, train_loader, val_loader, epochs=10, device="cuda"):
    model.to(device)
    
    # 🔥 UPDATED parameters applied here
    criterion = OCSoftmax(feat_dim=128, r_real=0.75, r_fake=0.15, fake_penalty=1.25).to(device)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=2e-5,
        weight_decay=4e-4
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # =========================
        # TRAIN
        # =========================
        model.train()
        criterion.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc="Training")

        for batch in loop:
            if len(batch) == 2:
                X, y = batch
            else:
                X, y, _ = batch

            X = X.to(device)
            y = y.long().to(device)

            optimizer.zero_grad()

            features = model(X)
            loss, scores = criterion(features, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        criterion.eval()
        val_loss = 0.0

        chunk_correct = 0
        chunk_total = 0
        chunk_false_bonafide = 0  
        chunk_false_spoof = 0     
        chunk_spoof_scores = []  
        chunk_labels = []

        file_scores_dict = {}
        file_labels = {}

        # 🔥 UPDATED: Mathematical midpoint of 0.75 and 0.15
        decision_threshold = 0.45

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                X, y, filenames = batch
                X = X.to(device)
                y = y.long().to(device)

                features = model(X)
                loss, scores = criterion(features, y)
                val_loss += loss.item()

                y_cpu = y.cpu()
                scores_cpu = scores.cpu()
                
                preds = (scores_cpu <= decision_threshold).long()
                spoof_scores = -scores_cpu 

                # CHUNK-LEVEL STATS
                chunk_correct += (preds == y_cpu).sum().item()
                chunk_total += y_cpu.size(0)

                chunk_false_bonafide += ((y_cpu == 1) & (preds == 0)).sum().item()
                chunk_false_spoof += ((y_cpu == 0) & (preds == 1)).sum().item()

                chunk_spoof_scores.extend(spoof_scores.numpy().tolist())
                chunk_labels.extend(y_cpu.numpy().tolist())

                # FILE-LEVEL STATS
                for i, fname in enumerate(filenames):
                    if fname not in file_scores_dict:
                        file_scores_dict[fname] = []
                        file_labels[fname] = int(y_cpu[i].item())

                    file_scores_dict[fname].append(float(scores_cpu[i].item()))

        val_loss /= len(val_loader)

        # CHUNK-LEVEL METRICS
        chunk_acc = chunk_correct / chunk_total
        chunk_false_bonafide_rate = chunk_false_bonafide / chunk_total
        chunk_false_spoof_rate = chunk_false_spoof / chunk_total
        chunk_eer = compute_eer(np.array(chunk_labels), np.array(chunk_spoof_scores))

        # FILE-LEVEL METRICS
        file_correct = 0
        file_total = 0
        file_false_bonafide = 0
        file_false_spoof = 0

        final_spoof_scores = []
        final_labels = []

        for fname in file_scores_dict:
            # 🔥 UPDATED: Switched from np.median to np.mean
            file_score = float(np.mean(file_scores_dict[fname]))
            
            pred = 1 if file_score <= decision_threshold else 0
            label = file_labels[fname]

            if pred == label:
                file_correct += 1
            if label == 1 and pred == 0:
                file_false_bonafide += 1
            if label == 0 and pred == 1:
                file_false_spoof += 1

            file_total += 1
            final_spoof_scores.append(-file_score)
            final_labels.append(label)

        file_acc = file_correct / file_total
        file_false_bonafide_rate = file_false_bonafide / file_total
        file_false_spoof_rate = file_false_spoof / file_total
        file_eer = compute_eer(np.array(final_labels), np.array(final_spoof_scores))

        # PRINT
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        print("\n--- Chunk-level / all voice batches ---")
        print(f"Chunk Acc:            {chunk_acc:.4f}")
        print(f"Chunk False Bonafide: {chunk_false_bonafide_rate * 100:.2f}%")
        print(f"Chunk False Spoof:    {chunk_false_spoof_rate * 100:.2f}%")
        # 🔥 UPDATED: Removed '* 100' so it prints the correct EER percentage
        print(f"Chunk EER:            {chunk_eer:.2f}%") 

        print("\n--- File-level ---")
        print(f"File Acc:             {file_acc:.4f}")
        print(f"File False Bonafide:  {file_false_bonafide_rate * 100:.2f}%")
        print(f"File False Spoof:     {file_false_spoof_rate * 100:.2f}%")
        # 🔥 UPDATED: Removed '* 100' so it prints the correct EER percentage
        print(f"File EER:             {file_eer:.2f}%")