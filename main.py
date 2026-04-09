import os
import sys
import uuid
import tempfile
import torch
import torchaudio
import torch.nn.functional as F
from transformers import WavLMModel

from model import SmallModel

# =========================
# CONFIG
# =========================
WAVLM_PATH = "./wavlm-base"
MODEL_PATH = "myTCN__asv5.pth"

TARGET_SR = 16000
CHUNK_LENGTH_SEC = 4
STEP_SIZE_SEC = 2

CHUNK_SAMPLES = CHUNK_LENGTH_SEC * TARGET_SR   # 64000
STEP_SAMPLES = STEP_SIZE_SEC * TARGET_SR       # 32000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# SCRAMBLER
# =========================
def purify_audio_batch_split_scrambler(batch_waveform, sr=16000, cutoff_freq=4000):
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft, device=batch_waveform.device)

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

def convert_to_flac(audio_path):
    import subprocess
    import tempfile

    tmp_flac = tempfile.NamedTemporaryFile(suffix=".flac", delete=False).name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ar", "16000",   # sample rate
        "-ac", "1",       # mono
        "-vn",
        tmp_flac
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(tmp_flac):
        raise RuntimeError("FFmpeg conversion failed")

    return tmp_flac

# =========================
# AUDIO LOADING
# =========================
def load_audio_any(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    err_torchaudio = None
    err_pydub = None

    try:
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr
    except Exception as e:
        err_torchaudio = e
        print(f"[torchaudio failed] {e}")

    try:
        from pydub import AudioSegment
        import numpy as np

        audio = AudioSegment.from_file(audio_path)
        sr = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width

        samples = np.array(audio.get_array_of_samples())

        if channels > 1:
            samples = samples.reshape((-1, channels)).T
        else:
            samples = samples.reshape((1, -1))

        max_val = float(1 << (8 * sample_width - 1))
        waveform = torch.tensor(samples, dtype=torch.float32) / max_val
        return waveform, sr

    except Exception as e:
        err_pydub = e

    raise RuntimeError(
        "Could not load audio file.\n"
        f"torchaudio error: {err_torchaudio}\n"
        f"pydub/ffmpeg error: {err_pydub}"
    )


def preprocess_audio(audio_path):
    waveform, sr = load_audio_any(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0)
    return audio


# =========================
# CHUNKING
# =========================
def chunk_audio(audio):
    num_samples = audio.shape[0]

    if num_samples < CHUNK_SAMPLES:
        pad_len = CHUNK_SAMPLES - num_samples
        audio = F.pad(audio, (0, pad_len))
    else:
        remainder = (num_samples - CHUNK_SAMPLES) % STEP_SAMPLES
        if remainder != 0:
            pad_len = STEP_SAMPLES - remainder
            audio = F.pad(audio, (0, pad_len))

    chunks = []
    for start in range(0, audio.shape[0] - CHUNK_SAMPLES + 1, STEP_SAMPLES):
        end = start + CHUNK_SAMPLES
        chunks.append(audio[start:end])

    return torch.stack(chunks)   # [num_chunks, 64000]


# =========================
# MODEL LOADING
# =========================
def load_wavlm():
    wavlm = WavLMModel.from_pretrained(WAVLM_PATH)
    for param in wavlm.parameters():
        param.requires_grad = False
    wavlm.to(device)
    wavlm.eval()
    return wavlm


def load_classifier():
    model = SmallModel() #myTCN()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# =========================
# FULL EMBEDDING EXTRACTION
# same style as flac_to_pt.py
# =========================
def extract_full_embeddings(wavlm, chunks):
    CHUNK_BATCH_SIZE = 64   # 🔥 adjust (8 if still OOM)

    all_embeddings = []

    total_chunks = chunks.shape[0]

    for i in range(0, total_chunks, CHUNK_BATCH_SIZE):
        batch = chunks[i:i + CHUNK_BATCH_SIZE].to(device)

        # 🔥 IMPORTANT: remove scrambler for inference
        input_chunks = batch

        mean = input_chunks.mean(dim=-1, keepdim=True)
        var = input_chunks.var(dim=-1, keepdim=True)
        input_values = (input_chunks - mean) / torch.sqrt(var + 1e-7)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = wavlm(input_values)
                    emb = outputs.last_hidden_state
            else:
                outputs = wavlm(input_values)
                emb = outputs.last_hidden_state

        all_embeddings.append(emb.cpu())   # 🔥 move to CPU immediately

        # 🔥 free GPU memory
        del batch, input_values, outputs, emb
        torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)   # [N, T, 768]

# =========================
# TEMP SAVE
# =========================
def save_embeddings_tmp(embeddings, source_audio_path):
    tmp_dir = os.path.join(tempfile.gettempdir(), "wavlm_tmp_embeddings")
    os.makedirs(tmp_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(source_audio_path))[0]
    unique_name = f"{base}_{uuid.uuid4().hex[:8]}.pt"
    save_path = os.path.join(tmp_dir, unique_name)

    torch.save(
        {
            "source_audio": source_audio_path,
            "shape": tuple(embeddings.shape),
            "embeddings": embeddings.cpu()
        },
        save_path
    )

    return save_path


# =========================
# PREDICT USING SAVED TMP EMBEDDINGS
# =========================
def predict_from_embeddings(classifier, embeddings):
    with torch.no_grad():
        logits = classifier(embeddings.to(device)).squeeze(1)   # [num_chunks]
        probs = torch.sigmoid(logits)

        # same file-level aggregation style as your training validation
        file_logit = torch.median(logits)
        file_prob = torch.sigmoid(file_logit).item()

    pred_label = "spoof" if file_prob > 0.5 else "bonafide"

    return {
        "chunk_probs": probs.detach().cpu().tolist(),
        "file_logit": float(file_logit.detach().cpu().item()),
        "spoof_probability": float(file_prob),
        "prediction": pred_label
    }


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"Using device: {device}")

    print("Loading WavLM...")
    wavlm = load_wavlm()

    print("Loading classifier...")
    classifier = load_classifier()

    print(f"Reading audio: {audio_path}")
    # convert everything → clean FLAC first
    converted_path = convert_to_flac(audio_path)
    print(f"Converted → {converted_path}")

    audio = preprocess_audio(converted_path)

    print(f"Duration (sec): {audio.shape[0] / TARGET_SR:.3f}")

    chunks = chunk_audio(audio)
    print(f"Chunk tensor shape: {tuple(chunks.shape)}")

    print("Extracting full embeddings...")
    embeddings = extract_full_embeddings(wavlm, chunks)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print("This is [num_chunks, T, 768]")

    tmp_path = save_embeddings_tmp(embeddings, audio_path)
    print(f"Saved tmp embeddings to: {tmp_path}")

    # optional reload from tmp to verify
    saved = torch.load(tmp_path, map_location="cpu", weights_only=False)
    embeddings_reloaded = saved["embeddings"]
    print(f"Reloaded tmp embeddings shape: {tuple(embeddings_reloaded.shape)}")

    result = predict_from_embeddings(classifier, embeddings_reloaded)

    print("\n===== RESULT =====")
    print(f"File: {audio_path}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Spoof probability: {result['spoof_probability']:.6f}")
    print(f"Chunks used: {embeddings_reloaded.shape[0]}")
    print(f"Frames per chunk (T): {embeddings_reloaded.shape[1]}")
    print(f"Embedding dim: {embeddings_reloaded.shape[2]}")


if __name__ == "__main__":
    main()