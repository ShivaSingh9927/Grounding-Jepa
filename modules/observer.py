import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class IJepaObserver(nn.Module):
    """
    Observer block using I-JEPA Vision Backbone from Hugging Face.
    Produces patch-level embeddings preserving spatial continuity.
    """
    def __init__(self, model_id="jmtzt/ijepa_vitg16_22k", device='cuda'):
        super().__init__()
        print(f"Loading I-JEPA Observer: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Determine embedding dimension from config
        self.embed_dim = self.model.config.hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: B, 3, 224, 224
        Output: B, N, D (N is patches, D is embed_dim)
        """
        # Note: AutoModel for I-JEPA expects pixels. 
        # If x is already a tensor [B, 3, 224, 224], we might need to handle it.
        # But if we use the processor earlier, we can pass it here.
        # For Grounding-JEPA, we expect patches.
        
        # The transformers I-JEPA model returns last_hidden_state [B, N, D]
        # where N includes the CLS token usually.
        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state
        
        # If there is a CLS token or similar, we might want to exclude it 
        # to keep the grid structure for C-Abstractor.
        # For jmtzt/ijepa_vitg16_22k, let's check if it has a CLS token.
        # Usually it's at index 0.
        if features.shape[1] == 197: # 14x14 + 1
            return features[:, 1:, :] 
        elif features.shape[1] == 257: # 16x16 + 1
            return features[:, 1:, :]
            
        return features
