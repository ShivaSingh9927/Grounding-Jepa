import torch
import torch.nn as nn
import math


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute 2D rotary frequency tensor for spatial awareness.
    
    Args:
        dim: Feature dimension
        height: Grid height (e.g., 14 for 224x224 with patch_size=16)
        width: Grid width
        theta: RoPE base theta
    
    Returns:
        freqs_cis: [H*W, Dim] complex tensor
    """
    y_vals = torch.arange(height, dtype=torch.float32)
    x_vals = torch.arange(width, dtype=torch.float32)
    
    y_emb = y_vals.unsqueeze(1).expand(height, width)
    x_emb = x_vals.unsqueeze(0).expand(height, width)
    
    y_emb = y_emb.flatten(0, 1)
    x_emb = x_emb.flatten(0, 1)
    
    dim_half = dim // 2
    theta_i = 1.0 / (theta ** (torch.arange(0, dim_half, 2, dtype=torch.float32) / dim_half))
    
    y_freqs = torch.outer(y_emb, theta_i)
    x_freqs = torch.outer(x_emb, theta_i)
    
    freqs = torch.cat([x_freqs, y_freqs], dim=-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rope_2d(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D RoPE to input tensor.
    
    Args:
        x: [Batch, Seq, Dim] - query or key tensor
        freqs_cis: [Seq, Dim] - precomputed rotary frequencies
    
    Returns:
        x_rotated: [Batch, Seq, Dim] with RoPE applied
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).expand(x.shape[0], -1, -1)
    x_rotated_complex = x_complex * freqs_cis
    x_rotated = torch.view_as_real(x_rotated_complex).flatten(-2).to(x.dtype)
    return x_rotated


class CAbstractor(nn.Module):
    """
    Convolutional Abstractor (C-Abstractor) for connecting visual features to LLM.
    Reduces the spatial grid (e.g., from 14x14 to 8x8) while preserving local context.
    
    Ablation: Optional 2D RoPE for spatial awareness.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 2048,
                 use_rope: bool = False, rope_theta: float = 10000.0):
        super().__init__()
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_rope:
            self.q_proj = nn.Linear(in_channels, hidden_dim)
            self.k_proj = nn.Linear(in_channels, hidden_dim)
            self.v_proj = nn.Linear(in_channels, hidden_dim)
            self.v_up_proj = nn.Linear(hidden_dim, out_channels)
            self.out_proj = nn.Linear(hidden_dim, out_channels)
            
            self.rope_freqs = None
            self.rope_grid_size = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_channels)
            )
    
    def _init_rope_freqs(self, grid_h: int, grid_w: int):
        if self.rope_freqs is None or self.rope_grid_size != (grid_h, grid_w):
            self.rope_freqs = precompute_freqs_cis_2d(
                self.q_proj.out_features, grid_h, grid_w, self.rope_theta
            ).to(next(self.parameters()).device)
            self.rope_grid_size = (grid_h, grid_w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch-level embeddings from I-JEPA [Batch, NumPatches, Channels]
        Returns:
            Projected visual tokens [Batch, 64, OutChannels]
        """
        B, L, C = x.shape
        grid_size = int(L ** 0.5)
        if grid_size * grid_size != L:
            raise ValueError(f"Input patches count {L} is not a perfect square.")
        
        x = x.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        x = self.pool(x)
        x = self.conv_block(x)
        x = x.flatten(2).transpose(1, 2)
        
        if self.use_rope:
            rope_h, rope_w = 8, 8
            self._init_rope_freqs(rope_h, rope_w)
            
            q = self.q_proj(x)
            k = self.k_proj(x)
            
            q = apply_rope_2d(q, self.rope_freqs)
            k = apply_rope_2d(k, self.rope_freqs)
            
            v = self.v_proj(x)
            v = self.v_up_proj(v)
            x = q * k
            x = self.out_proj(x)
            x = nn.functional.gelu(x + v)
        else:
            x = self.mlp(x)
        
        return x


class CAbstractorWithFinalProj(nn.Module):
    """
    Full C-Abstractor with optional RoPE and final projection to LLM dim.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 2048,
                 use_rope: bool = False, rope_theta: float = 10000.0):
        super().__init__()
        self.use_rope = use_rope
        
        if use_rope:
            self.bridge = CAbstractor(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=hidden_dim,
                use_rope=True,
                rope_theta=rope_theta
            )
        else:
            self.bridge = CAbstractor(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=hidden_dim,
                use_rope=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bridge(x)
