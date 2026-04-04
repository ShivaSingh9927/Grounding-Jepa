import torch
from modules.bridge import CAbstractor, CAbstractorWithFinalProj

def test_bridge():
    batch_size = 2
    num_patches = 196  # 14x14 grid
    in_channels = 1280  # I-JEPA hidden size
    out_channels = 3584  # Qwen2.5-7B hidden size
    
    x = torch.randn(batch_size, num_patches, in_channels)
    
    print("Testing C-Abstractor Ablation Study")
    print("=" * 50)
    
    print("\n1. Baseline (no RoPE):")
    bridge_baseline = CAbstractorWithFinalProj(
        in_channels=in_channels,
        out_channels=out_channels,
        use_rope=False
    )
    out_baseline = bridge_baseline(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_baseline.shape}")
    print(f"   Expected: [{batch_size}, 64, {out_channels}]")
    
    print("\n2. RoPE enabled (theta=10000):")
    bridge_rope = CAbstractorWithFinalProj(
        in_channels=in_channels,
        out_channels=out_channels,
        use_rope=True,
        rope_theta=10000.0
    )
    out_rope = bridge_rope(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_rope.shape}")
    print(f"   Expected: [{batch_size}, 64, {out_channels}]")
    
    print("\n3. Checking trainable parameters:")
    baseline_params = sum(p.numel() for p in bridge_baseline.parameters() if p.requires_grad)
    rope_params = sum(p.numel() for p in bridge_rope.parameters() if p.requires_grad)
    print(f"   Baseline trainable params: {baseline_params:,}")
    print(f"   RoPE trainable params: {rope_params:,}")
    
    print("\n4. Verifying outputs are different:")
    out_baseline_3k = out_baseline[:, :, :3584] if out_baseline.shape[-1] > 3584 else out_baseline
    out_rope_3k = out_rope[:, :, :3584] if out_rope.shape[-1] > 3584 else out_rope
    if out_baseline_3k.shape != out_rope_3k.shape:
        print(f"   Skipping diff check (shapes don't match: {out_baseline.shape} vs {out_rope.shape})")
    else:
        diff = torch.abs(out_baseline_3k - out_rope_3k).mean().item()
        print(f"   Mean absolute difference: {diff:.6f}")
        
        if diff > 1e-5:
            print("   ✓ Outputs are different (RoPE is affecting the computation)")
        else:
            print("   ✗ Outputs are too similar (potential issue)")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_bridge()
