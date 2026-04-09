import os
import torch
import torchaudio
import csv
from tqdm import tqdm

# Import the BATCHED loader and processor from the previous script
from inference_test import load_aasist_model, process_audio_batch

# --- Configuration ---
TSV_PATH = "./res/classification/ASVspoof5.train.tsv"
AUDIO_DIR = "./test"
# Changed output names so you can compare with the un-purified baseline
OUTPUT_SCORES_CSV = "./purified_scrambler_evaluation_scores.csv"
OUTPUT_STATS_TXT = "./purified_scrambler_evaluation_stats.txt"
BATCH_SIZE = 16  # Adjust based on your GPU VRAM

def purify_audio_batch_griffin_lim(batch_waveform):
    """
    Acts as an Information Bottleneck. 
    Takes a batched tensor [Batch, Time] on the GPU.
    Converts audio to a Spectrogram (destroying phase/adversarial noise)
    and reconstructs it using the Griffin-Lim algorithm simultaneously.
    """
    n_fft = 400
    hop_length = 160
    
    # 1. Setup the Spectrogram transformation on the GPU
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, 
        hop_length=hop_length, 
        power=2.0 # Power=2 gets the magnitude squared
    ).to(batch_waveform.device)
    
    # 2. Setup the Griffin-Lim reconstruction on the GPU
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, 
        hop_length=hop_length, 
        power=2.0,
        n_iter=32 # 32 iterations is the sweet spot for speed vs quality
    ).to(batch_waveform.device)
    
    # 3. The Bottleneck: Forward and Backward
    with torch.no_grad():
        # Strip phase data
        spec = spectrogram_transform(batch_waveform)
        # Rebuild audio from scratch mathematically
        purified_batch = griffin_lim(spec)
        
    # Ensure length matches original (Griffin-Lim can alter length slightly)
    original_len = batch_waveform.shape[1]
    purified_len = purified_batch.shape[1]
    
    if purified_len > original_len:
        purified_batch = purified_batch[:, :original_len]
    elif purified_len < original_len:
        padding = original_len - purified_len
        purified_batch = torch.nn.functional.pad(purified_batch, (0, padding))
        
    return purified_batch

def purify_audio_batch_gentle(batch_waveform):
    """
    A gentle Information Bottleneck.
    Uses 8-bit Mu-Law Quantization to crush adversarial amplitude gradients,
    and a Low-Pass filter to strip high-frequency adversarial static.
    Leaves human phase data 100% intact.
    """
    # 1. Setup Mu-law Quantization (256 channels = 8-bit audio)
    encode = torchaudio.transforms.MuLawEncoding(quantization_channels=256).to(batch_waveform.device)
    decode = torchaudio.transforms.MuLawDecoding(quantization_channels=256).to(batch_waveform.device)
    
    with torch.no_grad():
        # Step A: Crush the amplitude to 8-bit (Destroys micro-gradients)
        quantized_batch = encode(batch_waveform)
        
        # Step B: Restore it back to floating point so the model can read it
        restored_batch = decode(quantized_batch)
        
        # Step C: Apply a Low-Pass filter at 7000Hz (Chops off hidden high-frequency attacks)
        # Note: torchaudio.functional applies to the whole batch instantly
        purified_batch = torchaudio.functional.lowpass_biquad(
            restored_batch, 
            sample_rate=16000, 
            cutoff_freq=7000.0
        )
        
    return purified_batch


def purify_audio_batch_spectral_gating(batch_waveform):
    """
    Strips background noise (Acoustic Camouflage) while keeping Phase intact.
    Calculates the noise floor from the first ~100ms of audio and subtracts it.
    """
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft).to(batch_waveform.device)
    
    with torch.no_grad():
        # 1. Convert to Complex Spectrogram
        # return_complex=True is crucial: it preserves the phase data!
        stft = torch.stft(
            batch_waveform, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            window=window
        )
        
        # 2. Separate Magnitude (Volume) and Phase (Timing/Identity)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # 3. Estimate the Noise Profile
        # We assume the first 10 frames (~100ms) contain the baseline background noise
        noise_profile = torch.mean(magnitude[:, :, :10], dim=2, keepdim=True)
        
        # 4. Spectral Subtraction
        # Subtract the noise from the magnitude. The '1.5' is an aggression multiplier.
        # You can tweak this: 1.0 is gentle, 2.0 is highly aggressive denoising.
        clean_magnitude = magnitude - (noise_profile * 1.5)
        
        # Prevent mathematical errors by flooring negative values to a tiny number
        clean_magnitude = torch.max(clean_magnitude, torch.full_like(clean_magnitude, 0.01))
        
        # 5. Recombine the Cleaned Magnitude with the ORIGINAL Phase
        # This is why real humans won't sound robotic!
        clean_stft = clean_magnitude * torch.exp(1j * phase)
        
        # 6. Reconstruct the audio waveform
        clean_waveform = torch.istft(
            clean_stft, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            length=batch_waveform.shape[1]
        )
                                     
    return clean_waveform

def purify_audio_batch_split_scrambler(batch_waveform, sr=16000, cutoff_freq=4000):
    """
    The ultimate compromise:
    Preserves perfect phase in the low frequencies (keeps humans sounding human).
    Brutally randomizes phase in the high frequencies (destroys deepfake vocoder artifacts).
    """
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft).to(batch_waveform.device)
    
    with torch.no_grad():
        # 1. Convert to Complex Spectrogram
        stft = torch.stft(
            batch_waveform, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            window=window
        )
        
        magnitude = torch.abs(stft)
        original_phase = torch.angle(stft)
        
        # 2. Calculate where the "Human" audio ends and the "Air/Artifacts" begin
        # Nyquist frequency is exactly half the sample rate (8000 Hz)
        nyquist = sr / 2
        # Calculate which frequency bin corresponds to our 4000 Hz cutoff
        cutoff_bin = int((cutoff_freq / nyquist) * (n_fft // 2 + 1))
        
        # 3. Create a totally chaotic, random phase tensor
        # We generate random angles between -Pi and Pi
        chaotic_phase = (torch.rand_like(original_phase) * 2 * torch.pi) - torch.pi
        
        # 4. The Frankenstein Merge
        # Start with a copy of the perfect, original phase
        frankenstein_phase = original_phase.clone()
        # Overwrite ONLY the high frequencies with the chaotic, spoof-destroying phase
        frankenstein_phase[:, cutoff_bin:, :] = chaotic_phase[:, cutoff_bin:, :]
        
        # 5. Recombine the original Magnitude with our new Frankenstein Phase
        # torch.polar is the mathematically safest way to combine magnitude + phase in PyTorch
        scrambled_stft = torch.polar(magnitude, frankenstein_phase)
        
        # 6. Reconstruct the audio waveform
        purified_waveform = torch.istft(
            scrambled_stft, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            length=batch_waveform.shape[1]
        )
                                     
    return purified_waveform

def load_ground_truth(tsv_path):
    ground_truth = {}
    print(f"Loading ground truth from {tsv_path}...")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                file_id = parts[1]      
                label = parts[8]        
                ground_truth[file_id] = label
    print(f"Loaded {len(ground_truth)} labels.")
    return ground_truth

def evaluate_dataset():
    # 1. Load the model and get the active device (GPU/CPU)
    model, device = load_aasist_model()
    
    # 2. Load the ground truth labels
    ground_truth = load_ground_truth(TSV_PATH)
    
    # 3. Find all .flac files and filter them
    all_flac_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.flac')]
    
    valid_paths = []
    missing_labels = 0
    
    # Pre-filter files
    for filename in all_flac_files:
        file_id = filename.replace('.flac', '')
        if file_id in ground_truth:
            valid_paths.append(os.path.join(AUDIO_DIR, filename))
        else:
            missing_labels += 1
            
    print(f"Found {len(valid_paths)} valid .flac files to evaluate in batches of {BATCH_SIZE}.")
    
    # Statistics trackers
    stats = {
        "total": 0,
        "correct": 0,
        "true_bonafide": 0,   
        "false_bonafide": 0,  
        "true_spoof": 0,      
        "false_spoof": 0,     
        "missing_labels": missing_labels
    }
    
    results_data = []

    # 4. Batched Processing Loop
    for i in tqdm(range(0, len(valid_paths), BATCH_SIZE), desc="Purifying & Evaluating Batches"):
        batch_paths = valid_paths[i : i + BATCH_SIZE]
        
        # Process the raw audio into a batched GPU tensor
        batch_tensor, successful_paths = process_audio_batch(batch_paths, device)
        
        if batch_tensor is None:
            continue
            
        with torch.no_grad():
            # --- THE MAGIC HAPPENS HERE ---
            # Purify the entire batch on the GPU
            purified_batch_tensor = purify_audio_batch_split_scrambler(batch_tensor)
            
            # Pass the PURIFIED batch through the model
            _, outputs = model(purified_batch_tensor)
            # ------------------------------
            
            # Unpack the batch results
            for j in range(len(successful_paths)):
                file_path = successful_paths[j]
                file_id = os.path.basename(file_path).replace('.flac', '')
                true_label = ground_truth[file_id]
                
                spoof_logit = outputs[j][0].item()
                bonafide_logit = outputs[j][1].item()
                
                # Prediction logic
                predicted_label = "bonafide" if bonafide_logit > spoof_logit else "spoof"
                is_correct = (predicted_label == true_label)
                
                # Update Statistics
                stats["total"] += 1
                if is_correct:
                    stats["correct"] += 1
                    
                if true_label == "bonafide" and predicted_label == "bonafide":
                    stats["true_bonafide"] += 1
                elif true_label == "spoof" and predicted_label == "bonafide":
                    stats["false_bonafide"] += 1
                elif true_label == "spoof" and predicted_label == "spoof":
                    stats["true_spoof"] += 1
                elif true_label == "bonafide" and predicted_label == "spoof":
                    stats["false_spoof"] += 1

                # Save data for the CSV
                results_data.append([
                    file_id, true_label, predicted_label, spoof_logit, bonafide_logit
                ])

    # 5. Save detailed scores to CSV
    with open(OUTPUT_SCORES_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FileID", "TrueLabel", "PredictedLabel", "SpoofLogit", "BonafideLogit"])
        writer.writerows(results_data)
        
    # 6. Calculate final metrics and save summary
    accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    
    report = (
        f"--- SCRAMBLER AASIST Evaluation on ASVspoof 5 ---\n"
        f"Total Evaluated: {stats['total']}\n"
        f"Overall Accuracy: {accuracy:.2f}%\n"
        f"\n--- Confusion Matrix ---\n"
        f"True Bonafide (Correctly accepted):  {stats['true_bonafide']}\n"
        f"False Bonafide (Missed deepfakes):   {stats['false_bonafide']}  <-- Check this vs baseline!\n"
        f"True Spoof (Correctly caught):       {stats['true_spoof']}\n"
        f"False Spoof (False alarms):          {stats['false_spoof']}     <-- This might go up slightly\n"
        f"\nMissing Labels in TSV: {stats['missing_labels']}\n"
    )
    
    print("\n" + report)
    
    with open(OUTPUT_STATS_TXT, 'w') as f:
        f.write(report)
        
    print(f"Detailed scores saved to: {OUTPUT_SCORES_CSV}")

if __name__ == "__main__":
    evaluate_dataset()