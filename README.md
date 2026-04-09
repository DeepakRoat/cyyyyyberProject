# Fine-tuning WavLM for Deepfake Voice Detection

This project provides a script to fine-tune the pre-trained WavLM model for detecting deepfake voices using transfer learning.

## Prerequisites

- Python 3.8+
- Conda environment with required packages

## Installation

1. Activate your conda environment:
   ```bash
   conda activate mlenv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Your dataset should be organized as follows:

```
dataset/
├── train/
│   ├── real/
│   │   ├── real_audio1.wav
│   │   └── ...
│   └── fake/
│       ├── fake_audio1.wav
│       └── ...
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

- Audio files should be in WAV format
- Sample rate: 16kHz (as required by WavLM)
- Labels: `real` (0) and `fake` (1)

## Usage

Run the fine-tuning script:

```bash
python fine_tune_wavlm.py --data_path /path/to/your/dataset --output_dir ./output
```

### Arguments

- `--model_path`: Path to the WavLM model (default: ./wavlm-base)
- `--data_path`: Path to your dataset folder (required)
- `--output_dir`: Directory to save the fine-tuned model (default: ./wavlm-deepfake)

## Training Details

- Model: WavLM-Base
- Task: Binary classification (real vs fake)
- Batch size: 4
- Learning rate: 3e-5
- Epochs: 5
- Metrics: Accuracy, Precision, Recall, F1-score

## Inference

After training, you can use the fine-tuned model for inference:

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa

# Load model and feature extractor
model = AutoModelForAudioClassification.from_pretrained("./wavlm-deepfake")
feature_extractor = AutoFeatureExtractor.from_pretrained("./wavlm-base")

# Load audio
audio, sr = librosa.load("path/to/audio.wav", sr=16000)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

# Predict
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()

print("Real" if predicted_class == 0 else "Fake")
```

## Troubleshooting

- **CUDA out of memory**: Reduce batch size in the script
- **Audio format issues**: Ensure all audio files are WAV at 16kHz
- **Dataset loading errors**: Check folder structure matches the expected format
- **Model loading issues**: Ensure the model path is correct and all files are present

## References

- [WavLM Paper](https://arxiv.org/abs/2110.13900)
- [Hugging Face Audio Classification Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification)