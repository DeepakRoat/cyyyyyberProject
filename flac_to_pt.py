import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel
import os
from tqdm import tqdm
import torchaudio
import torch.nn.functional as F
import h5py
import numpy as np

# --- 1. SETUP ---
local_model_path = "./wavlm-base"

# We only need the model now; we built a faster GPU native processor!
model = WavLMModel.from_pretrained(local_model_path)

for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("WavLM model loaded successfully and frozen!")

AUDIO_DIR  = "./train"
SAVE_DIR = "D:/embeddings_train"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SR = 16000

# --- CHUNKING PARAMETERS ---
CHUNK_LENGTH_SEC = 4
STEP_SIZE_SEC = 2
CHUNK_SAMPLES = CHUNK_LENGTH_SEC * TARGET_SR  # 64,000
STEP_SAMPLES = STEP_SIZE_SEC * TARGET_SR      # 32,000


# --- 2. THE PURIFIER ---
def purify_audio_batch_split_scrambler(batch_waveform, sr=16000, cutoff_freq=4000):
    """
    Preserves perfect phase in the low frequencies (0-4000 Hz).
    Brutally randomizes phase in the high frequencies (4000-8000 Hz).
    Runs entirely on the GPU.
    """
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft).to(batch_waveform.device)
    
    stft = torch.stft(
        batch_waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True, 
        window=window
    )
    
    magnitude = torch.abs(stft)
    original_phase = torch.angle(stft)
    
    nyquist = sr / 2
    cutoff_bin = int((cutoff_freq / nyquist) * (n_fft // 2 + 1))
    
    chaotic_phase = (torch.rand_like(original_phase) * 2 * torch.pi) - torch.pi
    
    frankenstein_phase = original_phase.clone()
    frankenstein_phase[:, cutoff_bin:, :] = chaotic_phase[:, cutoff_bin:, :]
    
    scrambled_stft = torch.polar(magnitude, frankenstein_phase)
    
    purified_waveform = torch.istft(
        scrambled_stft, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=window,
        length=batch_waveform.shape[1]
    )
                                 
    return purified_waveform


# --- HDF5 RESUME HELPER ---
def get_processed_files(save_dir):
    """Scans all existing .h5 files and returns a set of filenames already processed."""
    processed = set()
    for f in os.listdir(save_dir):
        if f.endswith(".h5"):
            try:
                with h5py.File(os.path.join(save_dir, f), 'r') as h5f:
                    processed.update(h5f.keys())
            except Exception as e:
                print(f"Warning: Could not read {f}. It might be corrupted.")
    return processed


# --- 3. CPU WORKER CLASS ---
class AudioChunkDataset(Dataset):
    def __init__(self, audio_dir, save_dir):
        self.audio_dir = audio_dir
        self.save_dir = save_dir
        
        all_files = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
        self.files = []
        
        # New Resume Logic: Checks inside the HDF5 databases!
        processed_files = get_processed_files(save_dir)
        for f in all_files:
            if f not in processed_files:
                self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.audio_dir, file)

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)

        audio = waveform.squeeze(0)
        num_samples = audio.shape[0]

        if num_samples < CHUNK_SAMPLES:
            pad_len = CHUNK_SAMPLES - num_samples
            audio = F.pad(audio, (0, pad_len))
        else:
            remainder = (num_samples - CHUNK_SAMPLES) % STEP_SAMPLES
            if remainder != 0:
                pad_len = STEP_SAMPLES - remainder
                audio = F.pad(audio, (0, pad_len))

        num_samples = audio.shape[0]
        chunks = []

        for start in range(0, num_samples - CHUNK_SAMPLES + 1, STEP_SAMPLES):
            end = start + CHUNK_SAMPLES
            chunks.append(audio[start:end])

        chunks_tensor = torch.stack(chunks)
        return file, chunks_tensor

# --- 4. MAIN EXECUTION ---
def main():
    dataset = AudioChunkDataset(AUDIO_DIR, SAVE_DIR)
    
    if len(dataset) == 0:
        print("All files are already processed! Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False, pin_memory=True)
    print(f"Starting extraction for {len(dataset)} files...")

    CHUNK_BATCH_SIZE = 768      #512
    
    # HDF5 Batching Setup
    H5_BATCH_LIMIT = 1500
    file_counter = 0
    h5_file_idx = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".h5")])
    current_h5_path = os.path.join(SAVE_DIR, f"wavlm_features_batch_{h5_file_idx}.h5")
    
    # Open the first HDF5 file
    h5f = h5py.File(current_h5_path, 'a')

    for file_tuple, chunks_tensor in tqdm(dataloader):
        filename = file_tuple[0]
        chunks = chunks_tensor.squeeze(0) 
        total_chunks = chunks.shape[0]
        
        chunk_embeddings = []

        for i in range(0, total_chunks, CHUNK_BATCH_SIZE):
            batch_chunks = chunks[i : i + CHUNK_BATCH_SIZE].to(device)
            scrambled_chunks = purify_audio_batch_split_scrambler(batch_chunks)
            
            mean = scrambled_chunks.mean(dim=-1, keepdim=True)
            var = scrambled_chunks.var(dim=-1, keepdim=True)
            input_values = (scrambled_chunks - mean) / torch.sqrt(var + 1e-7)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_values)  
                    emb = outputs.last_hidden_state
                    
                # Keep it in FP16 before moving to CPU to save RAM
                chunk_embeddings.append(emb.cpu().to(torch.float16))

        # Combine chunks and convert to a pure FP16 NumPy array for HDF5
        final_file_embedding = torch.cat(chunk_embeddings, dim=0).numpy()
        
        # Save directly into the HDF5 database with compression
        h5f.create_dataset(filename, data=final_file_embedding, compression="gzip", compression_opts=4)

        # Batch Management: Close file and open a new one every 1000 files
        file_counter += 1
        if file_counter % H5_BATCH_LIMIT == 0:
            h5f.close()
            h5_file_idx += 1
            current_h5_path = os.path.join(SAVE_DIR, f"wavlm_features_batch_{h5_file_idx}.h5")
            h5f = h5py.File(current_h5_path, 'a')

    # Close the final file when the loop finishes
    h5f.close()
    print("All chunked embeddings saved successfully to HDF5 databases!")

if __name__ == '__main__':
    main()