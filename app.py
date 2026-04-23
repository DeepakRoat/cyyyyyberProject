import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress transformers warnings BEFORE any imports
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import torch
import tempfile
import traceback

# DO NOT import main here to avoid transformers import chain error
# Instead, import lazily in init_models_on_startup()

# Global model functions and status
models_ready = False
init_error = None
initialize_models_fn = None
process_audio_file_fn = None

def init_models_on_startup():
    """Initialize models on startup with lazy imports"""
    global models_ready, init_error, initialize_models_fn, process_audio_file_fn
    try:
        print("Initializing models on startup...")
        # Lazy import from main AFTER transformers warnings are suppressed
        from main import initialize_models, process_audio_file
        initialize_models_fn = initialize_models
        process_audio_file_fn = process_audio_file
        
        # Now initialize
        initialize_models_fn()
        models_ready = True
        print("✓ Models initialized successfully")
        return True
    except Exception as e:
        init_error = str(e)
        print(f"✗ Failed to initialize models: {e}")
        traceback.print_exc()
        return False

def analyze_audio(audio_input):
    """
    Analyze audio file using WavLM + SmallModel on GPU.
    Input: audio file path or uploaded file
    """
    global models_ready, init_error, process_audio_file_fn
    
    if not models_ready:
        if init_error:
            return f"❌ Model initialization failed:\n{init_error}"
        return "❌ Models not initialized. Please refresh and try again."
    
    if audio_input is None:
        return "⚠️ Please upload an audio file."
    
    try:
        # audio_input is a file path string from Gradio
        result = process_audio_file_fn(audio_input)
        
        # Format output
        output = f"""
╔════════════════════════════════════════╗
║     AUDIO DEEPFAKE DETECTION RESULT    ║
╚════════════════════════════════════════╝

📁 File: {result['file']}

🎯 PREDICTION: {result['prediction']}

📊 CONFIDENCE SCORES:
  • Spoof Probability (Mean): {result['spoof_probability']:.4f} ({result['spoof_probability']*100:.2f}%)
  • Bonafide Probability: {1-result['spoof_probability']:.4f} ({(1-result['spoof_probability'])*100:.2f}%)

📈 ANALYSIS DETAILS:
  • Total Chunks Analyzed: {result['chunk_count']}
  • Frames per Chunk: {result['frames_per_chunk']}
  • Embedding Dimension: {result['embedding_dim']}
  • Mean Chunk Probability: {result['mean_prob']:.6f}
  • Min Chunk Prob: {min(result['chunk_probs']):.4f}
  • Max Chunk Prob: {max(result['chunk_probs']):.4f}
  • Std Dev: {(sum((p - result['mean_prob'])**2 for p in result['chunk_probs']) / len(result['chunk_probs']))**0.5:.4f}
"""
        return output
        
    except Exception as e:
        error_msg = f"❌ Error analyzing audio:\n{str(e)}\n\nDetails:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

# Initialize models at startup
print("=" * 50)
print("AUDIO DEEPFAKE DETECTOR - INITIALIZATION")
print("=" * 50)
init_models_on_startup()

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Textbox(label="Analysis Result", lines=20),
    title="🎙️ Audio Deepfake Detection System",
    description="Upload an audio file to detect if it's a real voice or AI-generated/spoofed audio using WavLM + SmallModel (GPU accelerated).",
    examples=[]
)

if __name__ == "__main__":
    print("=" * 50)
    print("Launching Gradio interface...")
    print("=" * 50)
    # To share with external network, set share=True
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)

