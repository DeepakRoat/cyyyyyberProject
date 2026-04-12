import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig

# Import your existing TCN from your unmodified model.py
from model import myTCN

class EndToEndModel(nn.Module):
    def __init__(self, wavlm_path="./wavlm-base"):
        super().__init__()
        
        # 1. Load the 4-Layer Truncated WavLM
        print("Loading and truncating WavLM to 4 Layers...")
        config = WavLMConfig.from_pretrained(wavlm_path)
        config.num_hidden_layers = 4
        self.wavlm = WavLMModel.from_pretrained(wavlm_path, config=config)
        
        # 2. Freeze WavLM entirely to save VRAM and compute
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        # 3. Attach your trainable TCN
        self.tcn = myTCN()

    def forward(self, x):
        """
        Expects raw audio shape: [Batch, Time]
        """
        # A. Normalize the raw audio 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-7)

        # B. Extract acoustic features using the frozen WavLM 
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.wavlm(x)
                # Because the model is truncated to 4 layers, this IS Layer 4!
                features = outputs.last_hidden_state  # Shape: [Batch, Time, 768]
                
        # C. Pass the features into the TCN (converted back to FP32 for stability)
        out = self.tcn(features.float())
        return out