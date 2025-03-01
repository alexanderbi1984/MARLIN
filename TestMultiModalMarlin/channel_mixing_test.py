import torch




def test_input_data():
    # Load a single batch from your DataLoader
    sample_batch = next(iter(dataloader))
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Print shapes and value ranges
    print(f"Mixed video shape: {mixed_video.shape}, range: [{mixed_video.min()}, {mixed_video.max()}]")
    print(f"RGB frames shape: {rgb_frames.shape}, range: [{rgb_frames.min()}, {rgb_frames.max()}]")
    print(f"Depth frames shape: {depth_frames.shape}, range: [{depth_frames.min()}, {depth_frames.max()}]")
    print(f"Thermal frames shape: {thermal_frames.shape}, range: [{thermal_frames.min()}, {thermal_frames.max()}]")

    # Visualize a single frame from each modality
    import matplotlib.pyplot as plt

    # Get a middle frame
    frame_idx = mixed_video.shape[2] // 2

    plt.figure(figsize=(15, 10))

    # Show mixed input channels
    plt.subplot(2, 2, 1)
    plt.title("Mixed Input (first 3 channels)")
    plt.imshow(mixed_video[0, :3, frame_idx].permute(1, 2, 0).cpu().numpy())

    # Show RGB
    plt.subplot(2, 2, 2)
    plt.title("RGB")
    plt.imshow(rgb_frames[0, :, frame_idx].permute(1, 2, 0).cpu().numpy())

    # Show Depth (single channel)
    plt.subplot(2, 2, 3)
    plt.title("Depth")
    plt.imshow(depth_frames[0, 0, frame_idx].cpu().numpy(), cmap='magma')

    # Show Thermal (single channel)
    plt.subplot(2, 2, 4)
    plt.title("Thermal")
    plt.imshow(thermal_frames[0, 0, frame_idx].cpu().numpy(), cmap='inferno')

    plt.tight_layout()
    plt.savefig("input_modalities.png")
    plt.close()


def inspect_channel_mixing():
    # Load a single batch
    sample_batch = next(iter(dataloader))
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Check if the mixed_video is actually a combination of the individual modalities
    # Extract a single frame for comparison
    frame_idx = mixed_video.shape[2] // 2
    mixed_frame = mixed_video[0, :, frame_idx]
    rgb_frame = rgb_frames[0, :, frame_idx]
    depth_frame = depth_frames[0, :, frame_idx]
    thermal_frame = thermal_frames[0, :, frame_idx]

    # Check if mixed_video channels match the individual modalities
    # Assuming the first 3 channels are RGB
    rgb_diff = torch.abs(mixed_frame[:3] - rgb_frame).mean().item()
    print(f"Average difference between mixed RGB and RGB frames: {rgb_diff}")

    # Assuming the 4th channel is depth
    if mixed_frame.shape[0] > 3:
        depth_diff = torch.abs(mixed_frame[3:4] - depth_frame).mean().item()
        print(f"Average difference between mixed depth and depth frames: {depth_diff}")

    # Assuming the 5th channel is thermal
    if mixed_frame.shape[0] > 4:
        thermal_diff = torch.abs(mixed_frame[4:5] - thermal_frame).mean().item()
        print(f"Average difference between mixed thermal and thermal frames: {thermal_diff}")

    # Visualize the channels of mixed_video
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    # Plot each channel of the mixed video
    num_channels = mixed_frame.shape[0]
    for i in range(num_channels):
        plt.subplot(2, 3, i + 1)
        plt.title(f"Channel {i}")
        plt.imshow(mixed_frame[i].cpu().numpy(), cmap='viridis')

    plt.tight_layout()
    plt.savefig("mixed_channels.png")
    plt.close()


def test_encoder_representations(model):
    # Register a hook to capture intermediate activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    # Register hooks at key points in the encoder
    model.encoder.register_forward_hook(hook_fn('encoder_output'))

    # Get a sample batch
    sample_batch = next(iter(dataloader))
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Take just one sample for simplicity
    mixed_video = mixed_video[:1]
    mask = mask[:1]

    # Forward pass through the encoder
    with torch.no_grad():
        encoded = model.encoder(mixed_video, mask)

    # Print shape of encoder output
    print(f"Encoder output shape: {encoded.shape}")
    print(f"Encoder output stats: min={encoded.min().item()}, max={encoded.max().item()}, mean={encoded.mean().item()}")

    # Visualize the first few dimensions of the encoder output
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.title("Encoder Output Feature Map")
    plt.imshow(activations['encoder_output'][0, :100, :100].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.savefig("encoder_features.png")
    plt.close()


def test_decoders(model):
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Take just one sample
    mixed_video = mixed_video[:1]
    mask = mask[:1]
    rgb_frames = rgb_frames[:1]
    depth_frames = depth_frames[:1]
    thermal_frames = thermal_frames[:1]

    # Create a highly recognizable pattern in each modality to test separation
    # Each modality will have a distinctive pattern
    frame_idx = mixed_video.shape[2] // 2

    # Create RGB pattern (horizontal stripes)
    rgb_frames[:, :, frame_idx, :, :] = 0
    for i in range(0, rgb_frames.shape[3], 20):
        rgb_frames[:, :, frame_idx, i:i + 10, :] = 1.0

    # Create depth pattern (vertical stripes)
    depth_frames[:, :, frame_idx, :, :] = 0
    for i in range(0, depth_frames.shape[4], 20):
        depth_frames[:, :, frame_idx, :, i:i + 10] = 1.0

    # Create thermal pattern (diagonal stripes)
    thermal_frames[:, :, frame_idx, :, :] = 0
    for i in range(0, min(thermal_frames.shape[3], thermal_frames.shape[4]), 20):
        for j in range(10):
            if i + j < thermal_frames.shape[3] and i + j < thermal_frames.shape[4]:
                thermal_frames[:, :, frame_idx, i + j, i + j] = 1.0

    # Create a new mixed video with these patterns
    # This part needs to match your mixing logic!
    test_mixed_video = torch.cat([rgb_frames, depth_frames, thermal_frames], dim=1)

    # Forward pass through the model
    with torch.no_grad():
        rgb_pred, thermal_pred, depth_pred = model(test_mixed_video, mask)

    # Check if the reconstructed outputs maintain the distinctive patterns
    import matplotlib.pyplot as plt

    # Ensure these match your actual unpatch_to_img functionality
    rgb_rec = model.rgb_decoder.unpatch_to_img(rgb_pred)
    depth_rec = model.depth_decoder.unpatch_to_img(depth_pred)
    thermal_rec = model.thermal_decoder.unpatch_to_img(thermal_pred)

    plt.figure(figsize=(15, 10))

    # Original inputs with patterns
    plt.subplot(3, 2, 1)
    plt.title("Original RGB (horizontal)")
    plt.imshow(rgb_frames[0, :, frame_idx].permute(1, 2, 0).cpu().numpy())

    plt.subplot(3, 2, 3)
    plt.title("Original Depth (vertical)")
    plt.imshow(depth_frames[0, 0, frame_idx].cpu().numpy(), cmap='magma')

    plt.subplot(3, 2, 5)
    plt.title("Original Thermal (diagonal)")
    plt.imshow(thermal_frames[0, 0, frame_idx].cpu().numpy(), cmap='inferno')

    # Reconstructed outputs
    plt.subplot(3, 2, 2)
    plt.title("Reconstructed RGB")
    plt.imshow(rgb_rec[0, :, frame_idx].permute(1, 2, 0).cpu().numpy())

    plt.subplot(3, 2, 4)
    plt.title("Reconstructed Depth")
    plt.imshow(depth_rec[0, 0, frame_idx].cpu().numpy(), cmap='magma')

    plt.subplot(3, 2, 6)
    plt.title("Reconstructed Thermal")
    plt.imshow(thermal_rec[0, 0, frame_idx].cpu().numpy(), cmap='inferno')

    plt.tight_layout()
    plt.savefig("decoder_separation_test.png")
    plt.close()


def test_cross_channel_leakage(model):
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Take just one sample
    mixed_video = mixed_video[:1]
    mask = mask[:1]

    # Create three test samples, each with only one modality present
    rgb_only = mixed_video.clone()
    depth_only = mixed_video.clone()
    thermal_only = mixed_video.clone()

    # Zero out other modalities (adjust channel indices based on your implementation)
    # Assuming RGB is first 3 channels, depth is 4th, thermal is 5th
    rgb_only[:, 3:, :, :, :] = 0  # Keep only RGB
    depth_only[:, :3, :, :, :] = 0  # Zero RGB
    depth_only[:, 4:, :, :, :] = 0  # Keep only depth
    thermal_only[:, :4, :, :, :] = 0  # Keep only thermal

    # Process each through the model
    with torch.no_grad():
        rgb_only_pred = model(rgb_only, mask)
        depth_only_pred = model(depth_only, mask)
        thermal_only_pred = model(thermal_only, mask)

    # Check if each modality's decoder primarily responds to its corresponding input
    # For example, with RGB-only input, the RGB decoder should have strongest response

    # Function to calculate activation strength
    def calc_activation_strength(pred):
        return torch.abs(pred).mean().item()

    # With RGB-only input
    rgb_strength_from_rgb = calc_activation_strength(rgb_only_pred[0])  # RGB decoder
    depth_strength_from_rgb = calc_activation_strength(rgb_only_pred[1])  # Depth decoder
    thermal_strength_from_rgb = calc_activation_strength(rgb_only_pred[2])  # Thermal decoder

    # With Depth-only input
    rgb_strength_from_depth = calc_activation_strength(depth_only_pred[0])
    depth_strength_from_depth = calc_activation_strength(depth_only_pred[1])
    thermal_strength_from_depth = calc_activation_strength(depth_only_pred[2])

    # With Thermal-only input
    rgb_strength_from_thermal = calc_activation_strength(thermal_only_pred[0])
    depth_strength_from_thermal = calc_activation_strength(thermal_only_pred[1])
    thermal_strength_from_thermal = calc_activation_strength(thermal_only_pred[2])

    # Print results
    print("\nCross-channel information leakage test:")
    print("\nWith RGB-only input:")
    print(f"  RGB decoder activation: {rgb_strength_from_rgb}")
    print(f"  Depth decoder activation: {depth_strength_from_rgb}")
    print(f"  Thermal decoder activation: {thermal_strength_from_rgb}")

    print("\nWith Depth-only input:")
    print(f"  RGB decoder activation: {rgb_strength_from_depth}")
    print(f"  Depth decoder activation: {depth_strength_from_depth}")
    print(f"  Thermal decoder activation: {thermal_strength_from_depth}")

    print("\nWith Thermal-only input:")
    print(f"  RGB decoder activation: {rgb_strength_from_thermal}")
    print(f"  Depth decoder activation: {depth_strength_from_thermal}")
    print(f"  Thermal decoder activation: {thermal_strength_from_thermal}")

    # Ideally, each decoder should respond strongest to its own modality


def test_reconstruction_with_masks(model):
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    mixed_video, _, rgb_frames, depth_frames, thermal_frames = sample_batch

    # Take just one sample
    mixed_video = mixed_video[:1]
    rgb_frames = rgb_frames[:1]
    depth_frames = depth_frames[:1]
    thermal_frames = thermal_frames[:1]

    # Create various masking patterns
    b, _, t, h, w = mixed_video.shape
    num_patches = (t // model.tubelet_size) * (h // model.patch_size) * (w // model.patch_size)

    mask_patterns = {
        "light_masking": torch.ones(b, num_patches, device=mixed_video.device).bool(),
        "heavy_masking": torch.zeros(b, num_patches, device=mixed_video.device).bool(),
        "checkerboard": torch.ones(b, num_patches, device=mixed_video.device).bool(),
        "modality_specific": torch.ones(b, num_patches, device=mixed_video.device).bool()
    }

    # Modify masks for different patterns
    # Light masking: mask 25% randomly
    indices = torch.randperm(num_patches)[:num_patches // 4]
    mask_patterns["light_masking"][:, indices] = False

    # Heavy masking: only 25% visible
    indices = torch.randperm(num_patches)[:num_patches // 4]
    mask_patterns["heavy_masking"][:, indices] = True

    # Checkerboard: alternating mask
    mask_patterns["checkerboard"][:, ::2] = False

    # Modality-specific: mask certain areas completely
    # This depends on how you're organizing your patches
    # For example, we could mask out the top half
    mask_patterns["modality_specific"][:, :num_patches // 2] = False

    # Test reconstruction with each mask
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, (mask_name, mask) in enumerate(mask_patterns.items()):
        # Forward pass
        with torch.no_grad():
            rgb_pred, thermal_pred, depth_pred = model(mixed_video, mask)

        # Create visualization arrays
        frame_idx = t // 2  # Middle frame

        # Original frames
        rgb_orig = rgb_frames[0, :, frame_idx].permute(1, 2, 0).cpu().numpy()
        depth_orig = depth_frames[0, 0, frame_idx].cpu().numpy()
        thermal_orig = thermal_frames[0, 0, frame_idx].cpu().numpy()

        # Get reconstructed frames (this will depend on your unpatch_to_img implementation)
        # You may need to adapt this based on your specific implementation
        b, n_masked, c = rgb_pred.shape
        rgb_full = torch.zeros(b, num_patches, c, device=rgb_pred.device)
        rgb_full[~mask] = rgb_pred

        rgb_rec = model.rgb_decoder.unpatch_to_img(rgb_full)[0, :, frame_idx].permute(1, 2, 0).cpu().numpy()

        # Similar for other modalities
        # ...

        # Plot
        axes[i, 0].set_title(f"{mask_name} - Mask")
        # Visualize mask (reshape to 2D)
        mask_vis = mask[0].reshape((t // model.tubelet_size),
                                   (h // model.patch_size),
                                   (w // model.patch_size))
        mask_vis = mask_vis[frame_idx // model.tubelet_size].cpu().numpy()
        axes[i, 0].imshow(mask_vis, cmap='gray')

        axes[i, 1].set_title("RGB")
        axes[i, 1].imshow(rgb_rec)

        axes[i, 2].set_title("Depth")
        # Plot depth_rec

        axes[i, 3].set_title("Thermal")
        # Plot thermal_rec

    plt.tight_layout()
    plt.savefig("mask_reconstruction_tests.png")
    plt.close()


def test_visualization_method(model):
    # Create a synthetic batch with known patterns
    b, c, t, h, w = 1, 5, 16, 224, 224  # Adjust based on your model

    # Create synthetic RGB, depth, and thermal frames
    rgb_frames = torch.zeros(b, 3, t, h, w)
    depth_frames = torch.zeros(b, 1, t, h, w)
    thermal_frames = torch.zeros(b, 1, t, h, w)

    # Add recognizable patterns
    # RGB: red, green, blue squares
    rgb_frames[0, 0, :, :h // 2, :w // 2] = 1.0  # Red in top-left
    rgb_frames[0, 1, :, :h // 2, w // 2:] = 1.0  # Green in top-right
    rgb_frames[0, 2, :, h // 2:, :w // 2] = 1.0  # Blue in bottom-left

    # Depth: diagonal gradient
    for i in range(h):
        for j in range(w):
            depth_frames[0, 0, :, i, j] = (i + j) / (h + w)

    # Thermal: radial pattern
    center_h, center_w = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            dist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
            thermal_frames[0, 0, :, i, j] = 1.0 - min(1.0, dist / (h // 2))

    # Create mixed video (adjust based on your mixing strategy)
    mixed_video = torch.cat([rgb_frames, depth_frames, thermal_frames], dim=1)

    # Create a basic mask (50% masked)
    num_patches = (t // model.tubelet_size) * (h // model.patch_size) * (w // model.patch_size)
    mask = torch.zeros(b, num_patches, dtype=torch.bool)
    mask[0, :num_patches // 2] = True

    # Use the model's _log_sample_reconstruction_image method
    try:
        model._log_sample_reconstruction_image((mixed_video, mask, rgb_frames, depth_frames, thermal_frames))
        print("Visualization method completed successfully")
    except Exception as e:
        print(f"Error in visualization method: {e}")


def run_all_tests(model, dataloader):
    print("Starting channel mixing tests")
    print("=============================")

    print("\nTest 1: Input Data Verification")
    test_input_data()

    print("\nTest 2: Channel Mixing Inspection")
    inspect_channel_mixing()

    print("\nTest 3: Encoder Representations")
    test_encoder_representations(model)

    print("\nTest 4: Decoder Channel Separation")
    test_decoders(model)

    print("\nTest 5: Cross-Channel Information Leakage")
    test_cross_channel_leakage(model)

    print("\nTest 6: Reconstruction with Different Masks")
    test_reconstruction_with_masks(model)

    print("\nTest 7: Visualization Method Test")
    test_visualization_method(model)

    print("\nAll tests completed!")

import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from model.marlin_multimodal import MultiModalMarlin
from dataset.bp4d_multimodal import BP4DMultiModalDataModule

from marlin_pytorch.util import read_yaml
from util.misc import load_official_pretrain_model
import os
import platform

parser = argparse.ArgumentParser("MULTIMODALMARLIN pretraining")
parser.add_argument("--config", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--official_pretrained", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    config_path = args.config
    data_path = args.data_dir
    resume_ckpt = args.resume
    config = read_yaml(config_path)

    batch_size = args.batch_size
    max_epochs = args.epochs
    num_workers = args.num_workers
    official_pretrained = args.official_pretrained

    model_name = config["model_name"]
    learning_rate = config["learning_rate"]["base"]
    warmup_lr = config["learning_rate"]["warmup"]
    min_lr = config["learning_rate"]["min"]
    warmup_epochs = config["learning_rate"]["warmup_epochs"]
    n_gpus = args.n_gpus
    img_size = config["img_size"]
    patch_size = config["patch_size"]
    clip_frames = config["clip_frames"]
    tubelet_size = config["tubelet_size"]
    mask_strategy = config["mask_strategy"]
    temporal_sample_rate = config["temporal_sample_rate"]
    mask_percentage_target = config["mask_percentage_target"]
    encoder_embed_dim = config["encoder"]["embed_dim"]
    encoder_depth = config["encoder"]["depth"]
    encoder_num_heads = config["encoder"]["num_heads"]
    decoder_embed_dim = config["decoder"]["embed_dim"]
    decoder_depth = config["decoder"]["depth"]
    decoder_num_heads = config["decoder"]["num_heads"]
    mlp_ratio = config["mlp_ratio"]
    qkv_bias = config["qkv_bias"]
    qk_scale = config["qk_scale"]
    drop_rate = config["drop_rate"]
    attn_drop_rate = config["attn_drop_rate"]
    norm_layer = config["norm_layer"]
    init_values = config["init_values"]
    optimizer_type = config["optimizer"]["type"]
    optimizer_eps = config["optimizer"]["eps"]
    optimizer_betas = config["optimizer"]["betas"]
    weight_decay = config["weight_decay"]
    adv_loss = config["adv_loss"]
    adv_weight = config["adv_weight"]
    gp_weight = config["gp_weight"]
    d_steps = config["d_steps"]
    g_steps = config["g_steps"]
    rgb_weight = config["rgb_weight"]
    thermal_weight = config["thermal_weight"]
    depth_weight = config["depth_weight"]
    # (f"the type of img_size is {type(img_size)} before init the model")

    total_batch_size = batch_size * n_gpus
    learning_rate = learning_rate * total_batch_size / 256
    warmup_lr = warmup_lr * total_batch_size / 256
    min_lr = min_lr * total_batch_size / 256

    dm = BP4DMultiModalDataModule(
        root_dir=data_path,
        batch_size=batch_size,
        clip_frames=clip_frames,
        temporal_sample_rate=temporal_sample_rate,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        mask_percentage_target=mask_percentage_target,
        mask_strategy=mask_strategy,
        num_workers=num_workers,
        take_train=None,
        take_val=None
    )
    dm.setup()

    # if adv_loss:
    #     from model.marlin import Marlin
    # else:
    #     raise NotImplementedError

    model = MultiModalMarlin(
        img_size=img_size,
        patch_size=patch_size,
        n_frames=clip_frames,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        norm_layer=norm_layer,
        init_values=init_values,
        tubelet_size=tubelet_size,
        optimizer_type=optimizer_type,
        optimizer_eps=optimizer_eps,
        optimizer_betas=optimizer_betas,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        warmup_lr=warmup_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        iter_per_epoch=len(dm.train_dataloader()),
        distributed=n_gpus > 1,
        d_steps=d_steps,
        g_steps=g_steps,
        adv_weight=adv_weight,
        gp_weight=gp_weight,
        rgb_weight=rgb_weight,
        thermal_weight=thermal_weight,
        depth_weight=depth_weight,
        name=model_name
    )

    run_all_tests(model, dm.train_dataloader())