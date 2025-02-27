from model.marlin_multimodal import MultiModalMarlin
import torch

# Create a simplified model with smaller dimensions
mini_model = MultiModalMarlin(
    img_size=224,  # Half the usual size
    patch_size=16,
    n_frames=16,  # Fewer frames
    encoder_embed_dim=768,  # Smaller embedding dimensions
    encoder_depth=12,  # Fewer transformer layers
    encoder_num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=8,
    tubelet_size=2
)

# Generate a small fake batch
batch_size = 2
mixed_video = torch.randn(batch_size, 3, 16, 224, 224)  # Adjust channel count if needed
mask = torch.ones(batch_size, (16 // 2) * (224 // 16) * (224 // 16)).bool()
mask[:, :10] = False  # Mask the first 10 patches for testing

# Test forward pass
with torch.no_grad():
    rgb_out, thermal_out, depth_out = mini_model(mixed_video, mask)

    print(f"RGB output shape: {rgb_out.shape}")
    print(f"Thermal output shape: {thermal_out.shape}")
    print(f"Depth output shape: {depth_out.shape}")