import os
import random
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from model import myTCN, train_model, SmallModel


def load_labels(protocol_file):
    """
    ASVspoof5.train.tsv-like format:
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
                label_map[filename] = 0.0
            elif label_str == "spoof":
                label_map[filename] = 1.0

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
        print(h5_path)
        with h5py.File(h5_path, "r") as h5f:
            for filename in h5f.keys():
                inventory[filename] = (h5_path, h5f[filename].shape)

    return inventory


def limit_test_filenames(test_filenames, train_filenames, seed=42):
    """
    Keep test count so that train:test ~= 0.7:0.3
    i.e. test <= (3/7) * train
    """
    max_test = int((3 / 7) * len(train_filenames))
    keep_n = min(len(test_filenames), max_test)

    test_filenames = list(test_filenames)
    rng = random.Random(seed)
    rng.shuffle(test_filenames)

    return test_filenames[:keep_n]


class H5ChunkDataset(Dataset):
    def __init__(self, filename_list, label_map, inventory, return_filename=False):
        self.filename_list = filename_list
        self.label_map = label_map
        self.inventory = inventory
        self.return_filename = return_filename
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

        if self.return_filename:
            return x, y, filename
        return x, y

    def __del__(self):
        for h5f in self._h5_cache.values():
            try:
                h5f.close()
            except Exception:
                pass


def main():
    use_previousWeights = True
    MODEL_PATH = "myTCN__asv5.pth"
    # =========================
    # PATHS
    # =========================
    TEST_H5_DIR = r"D:/embeddings_test"
    TRAIN_H5_DIR  = r"D:/embeddings_train"

    TRAIN_PROTOCOL_FILE = r"./res/classification/ASVspoof5.train.tsv"
    TEST_PROTOCOL_FILE  = r"./res/classification/ASVspoof5.dev.track_1.tsv"   # change if needed

    # =========================
    # HYPERPARAMS
    # =========================
    BATCH_SIZE = 128    #128
    EPOCHS = 2
    NUM_WORKERS = 6
    SEED = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # =========================
    # LOAD LABELS
    # =========================
    train_label_map = load_labels(TRAIN_PROTOCOL_FILE)
    test_label_map = load_labels(TEST_PROTOCOL_FILE)

    print(f"Loaded train labels for {len(train_label_map)} files")
    print(f"Loaded test labels for  {len(test_label_map)} files")

    # =========================
    # LOAD H5 FILES
    # =========================
    train_h5_files = get_all_h5_files(TRAIN_H5_DIR)
    test_h5_files = get_all_h5_files(TEST_H5_DIR)

    print(f"Found {len(train_h5_files)} train HDF5 files")
    print(f"Found {len(test_h5_files)} test HDF5 files")

    if len(train_h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {TRAIN_H5_DIR}")
    if len(test_h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {TEST_H5_DIR}")

    # =========================
    # INVENTORY
    # =========================
    train_inventory = scan_h5_inventory(train_h5_files)
    test_inventory = scan_h5_inventory(test_h5_files)

    print(f"Found {len(train_inventory)} embedded train filenames inside HDF5")
    print(f"Found {len(test_inventory)} embedded test filenames inside HDF5")

    # =========================
    # MATCH FILENAMES
    # =========================
    train_filenames = sorted(set(train_label_map.keys()) & set(train_inventory.keys()))
    test_filenames = sorted(set(test_label_map.keys()) & set(test_inventory.keys()))

    print(f"Matched train files: {len(train_filenames)}")
    print(f"Matched test files before limiting: {len(test_filenames)}")

    if len(train_filenames) == 0:
        raise ValueError(
            "No matching train filenames between protocol and train HDF5 keys."
        )
    if len(test_filenames) == 0:
        raise ValueError(
            "No matching test filenames between protocol and test HDF5 keys."
        )

    # =========================
    # LIMIT TEST COUNT
    # min(test, (7/3)*train)
    # =========================
    test_filenames = limit_test_filenames(
        test_filenames,
        train_filenames,
        seed=SEED
    )

    print(f"Train files used: {len(train_filenames)}")
    print(f"Test files used:  {len(test_filenames)}")
    print(f"Test cap = min(total_test, int((3/7)*train)) = {min(len(sorted(set(test_label_map.keys()) & set(test_inventory.keys()))), int((3/7) * len(train_filenames)))}")

    # =========================
    # DATASETS
    # =========================
    train_dataset = H5ChunkDataset(
        train_filenames,
        train_label_map,
        train_inventory,
        return_filename=False
    )

    val_dataset = H5ChunkDataset(
        test_filenames,
        test_label_map,
        test_inventory,
        return_filename=True
    )

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Test chunks:  {len(val_dataset)}")

    # =========================
    # DATALOADERS
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # =========================
    # SANITY CHECK
    # =========================
    X, y = next(iter(train_loader))
    print("Sample train batch X shape:", X.shape)
    print("Sample train batch y shape:", y.shape)

    Xv, yv, fnv = next(iter(val_loader))
    print("Sample test batch X shape:", Xv.shape)
    print("Sample test batch y shape:", yv.shape)
    print("Sample test filenames[0]:", fnv[0])

    # =========================
    # MODEL + TRAIN
    # =========================
    model = SmallModel() #myTCN()

    if use_previousWeights and os.path.exists(MODEL_PATH):
        print(f"Loading previous weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Training from scratch")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=device
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
