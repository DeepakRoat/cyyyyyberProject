import os
import random
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your existing training loop and wrapper model
from model import train_model
from e2e_model import EndToEndModel

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_labels(protocol_file):
    label_map = {}
    with open(protocol_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            filename = parts[1] + ".flac"
            label_str = parts[-2].lower()
            if label_str == "bonafide":
                label_map[filename] = 0.0
            elif label_str == "spoof":
                label_map[filename] = 1.0
    return label_map

def limit_test_filenames(test_filenames, train_filenames, seed=42):
    max_test = int((3 / 7) * len(train_filenames))
    keep_n = min(len(test_filenames), max_test)
    test_filenames = list(test_filenames)
    rng = random.Random(seed)
    rng.shuffle(test_filenames)
    return test_filenames[:keep_n]

# ==========================================
# DATASET (WITH DISK CACHING)
# ==========================================
class FlacChunkDataset(Dataset):
    def __init__(self, audio_dir, filenames, label_map, is_val=False, target_sr=16000, chunk_sec=4, step_sec=2, cache_path=None):
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.chunk_samples = chunk_sec * target_sr
        self.step_samples = step_sec * target_sr
        self.is_val = is_val
        self.index = []
        
        # 🔥 1. CHECK CACHE FIRST
        if cache_path and os.path.exists(cache_path):
            print(f"⚡ Loading instantly from cache: {cache_path}")
            self.index = torch.load(cache_path)
            return  # Skip the scanning entirely!

        # 🔥 2. IF NO CACHE, BUILD IT
        print(f"🔍 Cache not found. Scanning {len(filenames)} files in {audio_dir}...")
        for fname in tqdm(filenames, desc="Building Audio Index"):
            path = os.path.join(audio_dir, fname)
            if not os.path.exists(path):
                continue
                
            try:
                info = torchaudio.info(path)
                num_frames = info.num_frames
                sr = info.sample_rate
                
                ratio = sr / self.target_sr
                c_samp = int(self.chunk_samples * ratio)
                s_samp = int(self.step_samples * ratio)
                
                if num_frames < c_samp:
                    self.index.append((path, 0, num_frames, label_map[fname], fname, sr))
                else:
                    for start in range(0, num_frames - c_samp + 1, s_samp):
                        self.index.append((path, start, c_samp, label_map[fname], fname, sr))
            except Exception:
                pass

        # 🔥 3. SAVE TO CACHE FOR NEXT TIME
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.index, cache_path)
            print(f"💾 Saved new index cache to {cache_path}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, start, frames, label, fname, sr = self.index[idx]
        
        waveform, sample_rate = torchaudio.load(path, frame_offset=start, num_frames=frames)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
            
        audio = waveform.squeeze(0)
        num_samples = audio.shape[0]
        
        if num_samples < self.chunk_samples:
            pad_len = self.chunk_samples - num_samples
            audio = F.pad(audio, (0, pad_len))
            
        y = torch.tensor(label, dtype=torch.float32)
            
        if self.is_val:
            return audio, y, fname
        return audio, y

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    use_previousWeights = False
    MODEL_PATH = "myTCN__asv5_end_to_end.pth"
    WAVLM_PATH = "./wavlm-base"
    
    # Paths to raw audio
    TRAIN_AUDIO_DIR  = r"./train"  
    TEST_AUDIO_DIR   = r"./dev"   
    
    TRAIN_PROTOCOL_FILE = r"./res/classification/ASVspoof5.train.tsv"
    TEST_PROTOCOL_FILE  = r"./res/classification/ASVspoof5.dev.track_1.tsv"

    # 🔥 CACHE PATHS
    CACHE_DIR = "./index_cache"
    TRAIN_CACHE = os.path.join(CACHE_DIR, "train_chunks.pt")
    TEST_CACHE  = os.path.join(CACHE_DIR, "test_chunks.pt")

    BATCH_SIZE = 24     #16
    EPOCHS = 2
    NUM_WORKERS = 4
    SEED = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Labels & Filenames ---
    train_label_map = load_labels(TRAIN_PROTOCOL_FILE)
    test_label_map = load_labels(TEST_PROTOCOL_FILE)

    train_files_on_disk = set(os.listdir(TRAIN_AUDIO_DIR))
    test_files_on_disk = set(os.listdir(TEST_AUDIO_DIR))

    train_filenames = sorted(list(set(train_label_map.keys()) & train_files_on_disk))
    test_filenames = sorted(list(set(test_label_map.keys()) & test_files_on_disk))

    test_filenames = limit_test_filenames(test_filenames, train_filenames, seed=SEED)

    print(f"Train files matched: {len(train_filenames)}")
    print(f"Test files matched:  {len(test_filenames)}")

    # --- 2. Build Datasets (Now with caching!) ---
    train_dataset = FlacChunkDataset(
        TRAIN_AUDIO_DIR, train_filenames, train_label_map, 
        is_val=False, cache_path=TRAIN_CACHE
    )
    val_dataset = FlacChunkDataset(
        TEST_AUDIO_DIR, test_filenames, test_label_map, 
        is_val=True, cache_path=TEST_CACHE
    )

    print(f"Total Train Chunks: {len(train_dataset)}")
    print(f"Total Test Chunks:  {len(val_dataset)}")

    # --- 3. Build DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # --- 4. Initialize the End-to-End Model ---
    model = EndToEndModel(wavlm_path=WAVLM_PATH)
    
    if use_previousWeights and os.path.exists(MODEL_PATH):
        print(f"Loading previous TCN weights from {MODEL_PATH}")
        model.tcn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Training from scratch")

    # --- 5. Train ---
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=device
    )

    # Save only the TCN weights
    torch.save(model.tcn.state_dict(), MODEL_PATH)
    print(f"TCN Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()