import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

class JepaVisionWrapper(nn.Module):
    """
    Wrapper to make JEPA + Bridge compatible with Qwen2.5-VL's vision tower interface.
    """
    def __init__(self, observer, bridge):
        super().__init__()
        self.observer = observer
        self.bridge = bridge
        # Qwen2.5-VL uses 2x2 spatial pooling. Since our bridge outputs 8x8 tokens 
        # from a 16x16 raw grid (conceptually), we set spatial_merge_size=2.
        self.spatial_merge_size = 2 

    @property
    def dtype(self):
        # Get dtype from the first parameter of the bridge
        if hasattr(self.bridge, 'mlp') and len(self.bridge.mlp) > 0:
            return self.bridge.mlp[0].weight.dtype
        elif hasattr(self.bridge, 'q_proj'):
            return self.bridge.q_proj.weight.dtype
        else:
            return next(self.bridge.parameters()).dtype
        
    def forward(self, pixel_values, **kwargs):
        # pixel_values here is [B, 3, 224, 224] from our loader
        feat = self.observer(pixel_values) # [B, 196, 1280]
        tokens = self.bridge(feat) # [B, 64, 3584]
        
        # Qwen2.5-VL expects a flattened tensor of all tokens across the batch: [TotalTokens, HiddenSize]
        return tokens.flatten(0, 1) # [B*64, 3584]

class QwenReasoner(nn.Module):
    """
    Reasoner block using Qwen2.5-VL-7B with JEPA injection.
    """
    def __init__(self, model_id='Qwen/Qwen2.5-VL-7B-Instruct', device='cuda'):
        super().__init__()
        print(f"Loading Reasoner: {model_id}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def integrate_jepa(self, observer, bridge):
        """
        Replace Qwen's native vision tower with JEPA + Bridge.
        We must target the underlying VLModel's visual attribute.
        """
        wrapper = JepaVisionWrapper(observer, bridge)
        # Handle PeftModel wrapper if present
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        
        if hasattr(base_model, "model") and hasattr(base_model.model, "visual"):
            base_model.model.visual = wrapper
            print("JEPA Observer integrated into Qwen2.5-VLModel.")
        else:
            base_model.visual = wrapper
            print("JEPA Observer integrated into Reasoner model.")
        
    def forward(self, input_ids: torch.Tensor, images: torch.Tensor, image_grid_thw: torch.Tensor = None, attention_mask: torch.Tensor = None, labels: torch.Tensor = None, **kwargs):
        """
        Standard forward using the integrated vision tower.
        """
        if image_grid_thw is None:
            # Fallback to default for 224x224 if not provided
            batch_size = images.shape[0]
            image_grid_thw = torch.tensor([[1, 16, 16]] * batch_size, device=images.device, dtype=torch.long)
        
        return self.model(
            input_ids=input_ids,
            pixel_values=images,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate_answer(self, input_ids: torch.Tensor, images: torch.Tensor, image_grid_thw: torch.Tensor = None, **kwargs):
        """
        Standard generate using the integrated vision tower.
        """
        if image_grid_thw is None:
            batch_size = images.shape[0]
            image_grid_thw = torch.tensor([[1, 16, 16]] * batch_size, device=images.device, dtype=torch.long)
        
        # Ensure we use the correct special tokens for stopping
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=images,
            image_grid_thw=image_grid_thw,
            max_new_tokens=512,
            **kwargs
        )
        return outputs
