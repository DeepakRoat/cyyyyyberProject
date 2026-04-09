import torch
import torchaudio
import json
import os
import glob
from tqdm import tqdm

# 1. Import the model architecture directly from their folder
from aasist.models.AASIST import Model 

def load_aasist_model(config_path="./aasist/config/AASIST.conf", weights_path="./aasist/models/weights/AASIST.pth"):
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading architecture config...")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    print("Initializing empty model...")
    model = Model(config['model_config'])
    
    print("Injecting pre-trained weights...")
    # Load weights directly to the selected device
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    
    # Move model to GPU and lock into evaluation mode
    model = model.to(device)
    model.eval()
    print("AASIST model loaded successfully!\n")
    
    return model, device

def process_audio_batch(audio_paths, device, target_sr=16000, num_samples=64600):
    """
    Takes a list of audio file paths, processes them, and returns a single batched 
    tensor pushed to the correct device (GPU/CPU).
    """
    waveforms = []
    valid_paths = []
    
    for path in audio_paths:
        try:
            waveform, sr = torchaudio.load(path)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Resample if necessary
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
                
            # Squeeze out the channel dimension so shape is (Time,)
            waveform = waveform.squeeze(0)
            
            # Pad or truncate to exactly 64600 samples
            if waveform.shape[0] > num_samples:
                waveform = waveform[:num_samples]
            else:
                padding = num_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            waveforms.append(waveform)
            valid_paths.append(path)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    if not waveforms:
        return None, []
        
    # Stack the list of 1D tensors into a 2D tensor of shape (Batch, 64600)
    batch_tensor = torch.stack(waveforms)
    
    # Push the tensor to the GPU/CPU
    return batch_tensor.to(device), valid_paths


if __name__ == "__main__":
    # --- Configuration ---
    AUDIO_DIR = "./test"       # Directory containing your audio files
    BATCH_SIZE = 16            # Change this based on your GPU VRAM (e.g., 8, 16, 32, 64)
    
    # --- 1. Load the Model & Device ---
    model, device = load_aasist_model()
    
    # --- 2. Gather Audio Files ---
    # Gets all .flac and .wav files in the directory
    all_files = glob.glob(os.path.join(AUDIO_DIR, "*.flac")) + glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    
    if not all_files:
        print(f"No audio files found in {AUDIO_DIR}")
        exit()
        
    print(f"Found {len(all_files)} audio files to process in batches of {BATCH_SIZE}.")

    results = []

    # --- 3. Run Batched Inference ---
    # Chunk the list of files into batches
    for i in tqdm(range(0, len(all_files), BATCH_SIZE), desc="Processing Batches"):
        batch_paths = all_files[i : i + BATCH_SIZE]
        
        # Process the raw audio into a GPU tensor
        batch_tensor, valid_paths = process_audio_batch(batch_paths, device)
        
        if batch_tensor is None:
            continue
            
        with torch.no_grad():
            # Pass the whole batch through the model at once
            # output shape will be (Batch_Size, 2)
            _, outputs = model(batch_tensor)
            
            # Loop through the outputs to extract individual scores
            for j in range(len(valid_paths)):
                spoof_score = outputs[j][0].item()
                bonafide_score = outputs[j][1].item()
                
                prediction = "REAL" if bonafide_score > spoof_score else "FAKE"
                
                results.append({
                    "file": os.path.basename(valid_paths[j]),
                    "spoof_score": spoof_score,
                    "bonafide_score": bonafide_score,
                    "prediction": prediction
                })

    # --- 4. Display or Save Results ---
    print("\n--- Final Results ---")
    for res in results: # Showing just the first 10 for brevity in console
        print(f"{res['file']} -> {res['prediction']} (S: {res['spoof_score']:.2f}, B: {res['bonafide_score']:.2f})")