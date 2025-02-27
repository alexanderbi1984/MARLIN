from model.marlin_multimodal import MultiModalMarlin
import torch

# Create a simplified model with smaller dimensions
mini_model = MultiModalMarlin(
    img_size=112,  # Half the usual size
    patch_size=16,
    n_frames=8,  # Fewer frames
    encoder_embed_dim=192,  # Smaller embedding dimensions
    encoder_depth=2,  # Fewer transformer layers
    encoder_num_heads=4,
    decoder_embed_dim=128,
    decoder_depth=2,
    decoder_num_heads=4,
    tubelet_size=2
)

# Generate a small fake batch
batch_size = 2
mixed_video = torch.randn(batch_size, 3, 8, 112, 112)  # Adjust channel count if needed
mask = torch.ones(batch_size, (8 // 2) * (112 // 16) * (112 // 16)).bool()
mask[:, :10] = False  # Mask the first 10 patches for testing

# Test forward pass
with torch.no_grad():
    rgb_out, thermal_out, depth_out = mini_model(mixed_video, mask)

    print(f"RGB output shape: {rgb_out.shape}")
    print(f"Thermal output shape: {thermal_out.shape}")
    print(f"Depth output shape: {depth_out.shape}")