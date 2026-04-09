import os
import torch
import csv
from tqdm import tqdm

# Import the BATCHED loader and processor from the previous script
from inference_test import load_aasist_model, process_audio_batch

# --- Configuration ---
TSV_PATH = "./res/classification/ASVspoof5.train.tsv"
AUDIO_DIR = "./test"
OUTPUT_SCORES_CSV = "./aasist_evaluation_scores.csv"
OUTPUT_STATS_TXT = "./aasist_evaluation_stats.txt"
BATCH_SIZE = 16  # Adjust based on your GPU VRAM (e.g., 8, 16, 32, 64)

def load_ground_truth(tsv_path):
    """
    Parses the ASVspoof5 TSV file and returns a dictionary mapping
    filename (without extension) to its true label ('spoof' or 'bonafide').
    """
    ground_truth = {}
    print(f"Loading ground truth from {tsv_path}...")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                file_id = parts[1]      # e.g., T_0000000000
                label = parts[8]        # e.g., spoof or bonafide
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
    
    # Pre-filter files so we don't waste GPU time on unlabeled audio
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
        "true_bonafide": 0,   # Was bonafide, predicted bonafide
        "false_bonafide": 0,  # Was spoof, predicted bonafide (Dangerous!)
        "true_spoof": 0,      # Was spoof, predicted spoof
        "false_spoof": 0,     # Was bonafide, predicted spoof (Annoying)
        "missing_labels": missing_labels
    }
    
    results_data = []

    # 4. Batched Processing Loop
    for i in tqdm(range(0, len(valid_paths), BATCH_SIZE), desc="Evaluating Batches"):
        batch_paths = valid_paths[i : i + BATCH_SIZE]
        
        # Process the raw audio into a batched GPU tensor
        batch_tensor, successful_paths = process_audio_batch(batch_paths, device)
        
        if batch_tensor is None:
            continue
            
        with torch.no_grad():
            # Pass the entire batch through the model
            _, outputs = model(batch_tensor)
            
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
        f"--- AASIST Evaluation on ASVspoof 5 ---\n"
        f"Total Evaluated: {stats['total']}\n"
        f"Overall Accuracy: {accuracy:.2f}%\n"
        f"\n--- Confusion Matrix ---\n"
        f"True Bonafide (Correctly accepted):  {stats['true_bonafide']}\n"
        f"False Bonafide (Missed deepfakes):   {stats['false_bonafide']}  <-- The codec trap!\n"
        f"True Spoof (Correctly caught):       {stats['true_spoof']}\n"
        f"False Spoof (False alarms):          {stats['false_spoof']}\n"
        f"\nMissing Labels in TSV: {stats['missing_labels']}\n"
    )
    
    print("\n" + report)
    
    with open(OUTPUT_STATS_TXT, 'w') as f:
        f.write(report)
        
    print(f"Detailed scores saved to: {OUTPUT_SCORES_CSV}")

if __name__ == "__main__":
    evaluate_dataset()