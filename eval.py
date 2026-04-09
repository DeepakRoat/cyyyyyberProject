import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model import myTCN, SmallModel
from util import compute_eer


def load_labels(protocol_file):
    """
    ASVspoof5 TSV format:
    T_4850 T_0000000000 F - - - AC3 A05 spoof -
    T_3734 T_0000000011 F - - - - bonafide bonafide -

    filename -> second column + ".flac"
    label    -> second last column
    """
    label_map = {}

    with open(protocol_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            file_id = parts[1]
            filename = file_id + ".flac"
            label_str = parts[-2].lower()

            if label_str == "bonafide":
                label_map[filename] = 0
            elif label_str == "spoof":
                label_map[filename] = 1

    return label_map


def get_all_h5_files(h5_dir):
    h5_files = [
        os.path.join(h5_dir, f)
        for f in os.listdir(h5_dir)
        if f.endswith(".h5")
    ]
    h5_files.sort()
    return h5_files


def scan_h5_inventory(h5_paths):
    inventory = {}

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5f:
            for filename in h5f.keys():
                inventory[filename] = (h5_path, h5f[filename].shape)

    return inventory


class H5ChunkEvalDataset(Dataset):
    """
    Returns one chunk at a time:
        x, y, filename
    """
    def __init__(self, filename_list, label_map, inventory):
        self.filename_list = filename_list
        self.label_map = label_map
        self.inventory = inventory
        self.index = []
        self._h5_cache = {}

        for filename in self.filename_list:
            if filename not in self.inventory:
                continue
            if filename not in self.label_map:
                continue

            h5_path, shape = self.inventory[filename]

            # expected: [num_chunks, T, 768]
            if len(shape) != 3:
                print(f"Skipping malformed dataset: {filename}, shape={shape}")
                continue

            num_chunks, T, D = shape
            if D != 768:
                print(f"Skipping unexpected embedding dim: {filename}, shape={shape}")
                continue

            for chunk_idx in range(num_chunks):
                self.index.append((h5_path, filename, chunk_idx))

    def __len__(self):
        return len(self.index)

    def _get_h5(self, h5_path):
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, "r")
        return self._h5_cache[h5_path]

    def __getitem__(self, idx):
        h5_path, filename, chunk_idx = self.index[idx]
        h5f = self._get_h5(h5_path)

        chunk = h5f[filename][chunk_idx]   # [T, 768]
        label = self.label_map[filename]

        x = torch.tensor(chunk, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y, filename

    def __del__(self):
        for h5f in self._h5_cache.values():
            try:
                h5f.close()
            except Exception:
                pass


def evaluate_model(model, loader, device="cuda"):
    model.eval()
    model.to(device)

    file_probs = {}
    file_labels = {}

    with torch.no_grad():
        for X, y, filenames in loader:
            X = X.to(device)
            y = y.float().to(device)

            logits = model(X).squeeze(1)
            probs = torch.sigmoid(logits).cpu()

            for i, fname in enumerate(filenames):
                if fname not in file_probs:
                    file_probs[fname] = []
                    file_labels[fname] = int(y[i].item())

                file_probs[fname].append(float(probs[i].item()))

    correct = 0
    total = 0
    false_bonafide = 0  # y=1, pred=0
    false_spoof = 0     # y=0, pred=1

    final_scores = []
    final_labels = []

    for fname in file_probs:
        probs = file_probs[fname]

        # mean of probabilities
        file_prob = sum(probs) / len(probs)

        label = file_labels[fname]
        pred = 1 if file_prob > 0.5 else 0

        if pred == label:
            correct += 1

        if label == 1 and pred == 0:
            false_bonafide += 1

        if label == 0 and pred == 1:
            false_spoof += 1

        total += 1
        final_scores.append(file_prob)
        final_labels.append(label)

    acc = correct / total
    false_bonafide_rate = false_bonafide / total
    false_spoof_rate = false_spoof / total
    eer = compute_eer(np.array(final_labels), np.array(final_scores))

    print("\n===== FILE-LEVEL EVALUATION =====")
    print(f"Total files:        {total}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"False Bonafide %:   {false_bonafide_rate * 100:.2f}%")
    print(f"False Spoof %:      {false_spoof_rate * 100:.2f}%")
    print(f"EER:                {eer:.2f}%")

    return {
        "accuracy": acc,
        "false_bonafide_pct": false_bonafide_rate * 100,
        "false_spoof_pct": false_spoof_rate * 100,
        "eer_pct": eer,
    }


def main():
    # =========================
    # PATHS
    # =========================
    EVAL_H5_DIR = r"D:/embeddings_test"
    EVAL_PROTOCOL_FILE = r"./res/classification/ASVspoof5.dev.track_1.tsv"   # or dev/eval TSV
    MODEL_PATH = "myTCN__asv5.pth"

    # =========================
    # SETTINGS
    # =========================
    BATCH_SIZE = 128
    NUM_WORKERS = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    # labels
    label_map = load_labels(EVAL_PROTOCOL_FILE)
    print(f"Loaded labels for {len(label_map)} files")

    # h5
    h5_files = get_all_h5_files(EVAL_H5_DIR)
    print(f"Found {len(h5_files)} HDF5 files")

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {EVAL_H5_DIR}")

    inventory = scan_h5_inventory(h5_files)
    print(f"Found {len(inventory)} embedded filenames inside HDF5")

    usable_filenames = sorted(set(label_map.keys()) & set(inventory.keys()))
    print(f"Matched files: {len(usable_filenames)}")

    if len(usable_filenames) == 0:
        raise ValueError(
            "No matching filenames between protocol and HDF5 keys."
        )

    dataset = H5ChunkEvalDataset(
        usable_filenames,
        label_map,
        inventory
    )

    print(f"Total chunks: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # sanity check
    X, y, fn = next(iter(loader))
    print("Sample batch X shape:", X.shape)
    print("Sample batch y shape:", y.shape)
    print("Sample filename:", fn[0])

    # model
    model = SmallModel() #myTCN()
    print(f"Loading weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    evaluate_model(model, loader, device=device)


if __name__ == "__main__":
    main()
