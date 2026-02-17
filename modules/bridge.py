import torch
import torch.nn as nn

class CAbstractor(nn.Module):
    """
    Convolutional Abstractor (C-Abstractor) for connecting visual features to LLM.
    Reduces the spatial grid (e.g., from 32x32 to 16x16) while preserving local context.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 2048):
        super().__init__()
        # Use AdaptiveAvgPool2d to ensure we always get 8x8 grid (64 tokens)
        # regardless of the input patch count (e.g., 14x14 from I-JEPA 224x224)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # MLP for feature projection to LLM dimension
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch-level embeddings from I-JEPA [Batch, NumPatches, Channels]
        Returns:
            Projected visual tokens [Batch, 64, OutChannels]
        """
        B, L, C = x.shape
        grid_size = int(L**0.5)
        if grid_size * grid_size != L:
            raise ValueError(f"Input patches count {L} is not a perfect square.")
            
        # Reshape to grid for convolution/pooling
        x = x.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # Apply spatial reduction to 8x8
        x = self.pool(x)
        x = self.conv_block(x) # [B, C, 8, 8]
        
        # Flatten back to tokens
        x = x.flatten(2).transpose(1, 2) # [B, 64, C]
        
        # Project to LLM space
        x = self.mlp(x)
        
        return x
