import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class IJepaObserver(nn.Module):
    """
    Observer block using I-JEPA Vision Backbone from Hugging Face.
    Produces patch-level embeddings preserving spatial continuity.
    """
    def __init__(self, model_id="jmtzt/ijepa_vitg16_22k", device='cuda', checkpoint_path=None):
        super().__init__()
        print(f"Loading I-JEPA Observer: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        if checkpoint_path:
            self.load_meta_weights(checkpoint_path)
            
        # Determine embedding dimension from config
        self.embed_dim = self.model.config.hidden_size

    def load_meta_weights(self, path):
        print(f"Loading weights from Meta checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            state_dict = checkpoint
            
        # Mapping from Meta ViT to Transformers ViT
        mapping = {
            'patch_embed.proj.weight': 'embeddings.patch_embeddings.projection.weight',
            'patch_embed.proj.bias': 'embeddings.patch_embeddings.projection.bias',
            'pos_embed': 'embeddings.position_embeddings',
            'norm.weight': 'layernorm.weight',
            'norm.bias': 'layernorm.bias',
        }
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle standard common keys
            mapped_key = k
            if k in mapping:
                mapped_key = mapping[k]
            
            # Handle blocks
            if k.startswith('blocks.'):
                parts = k.split('.')
                block_idx = parts[1]
                rest = '.'.join(parts[2:])
                
                prefix = f'encoder.layer.{block_idx}'
                
                if rest == 'norm1.weight': mapped_key = f'{prefix}.layernorm_before.weight'
                elif rest == 'norm1.bias': mapped_key = f'{prefix}.layernorm_before.bias'
                elif rest == 'norm2.weight': mapped_key = f'{prefix}.layernorm_after.weight'
                elif rest == 'norm2.bias': mapped_key = f'{prefix}.layernorm_after.bias'
                elif rest == 'mlp.fc1.weight': mapped_key = f'{prefix}.intermediate.dense.weight'
                elif rest == 'mlp.fc1.bias': mapped_key = f'{prefix}.intermediate.dense.bias'
                elif rest == 'mlp.fc2.weight': mapped_key = f'{prefix}.output.dense.weight'
                elif rest == 'mlp.fc2.bias': mapped_key = f'{prefix}.output.dense.bias'
                elif rest == 'attn.proj.weight': mapped_key = f'{prefix}.attention.output.dense.weight'
                elif rest == 'attn.proj.bias': mapped_key = f'{prefix}.attention.output.dense.bias'
                elif rest == 'attn.qkv.weight':
                    # Meta uses [3*dim, dim], Qwen/ViT expects separate or handled
                    # We split qkv into query, key, value
                    dim = v.shape[1]
                    q, k, v_part = v.chunk(3, dim=0)
                    new_state_dict[f'{prefix}.attention.attention.query.weight'] = q
                    new_state_dict[f'{prefix}.attention.attention.key.weight'] = k
                    new_state_dict[f'{prefix}.attention.attention.value.weight'] = v_part
                    continue
                elif rest == 'attn.qkv.bias':
                    q, k, v_part = v.chunk(3, dim=0)
                    new_state_dict[f'{prefix}.attention.attention.query.bias'] = q
                    new_state_dict[f'{prefix}.attention.attention.key.bias'] = k
                    new_state_dict[f'{prefix}.attention.attention.value.bias'] = v_part
                    continue
            
            # Handle pos_embed shape (Meta has [1, 197, dim] or [1, 257, dim], HF ViT might want [1, 197, dim])
            if mapped_key == 'embeddings.position_embeddings' and v.shape[1] != self.model.embeddings.position_embeddings.shape[1]:
                print(f"Skipping pos_embed due to shape mismatch: {v.shape} vs {self.model.embeddings.position_embeddings.shape}")
                continue
                
            new_state_dict[mapped_key] = v
            
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Checkpoint loaded with message: {msg}")
        
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
