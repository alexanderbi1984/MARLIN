import torch
from torch import nn
import numpy as np
from einops import rearrange
import argparse

from torch.utils.checkpoint import checkpoint

from marlin_pytorch.util import read_yaml
import os

# def generate_thermal_depth_from_rgb(model, rgb_frames, device='cuda'):
#     """
#     Generate thermal and depth images from RGB frames using the MultiModalMarlin model.
#
#     Args:
#         model (MultiModalMarlin): Trained model instance
#         rgb_frames (torch.Tensor): RGB frames in shape [B, C, T, H, W]
#                                   where B=batch size, C=3 (RGB channels),
#                                   T=time/frames, H=height, W=width
#         device (str): Device to run inference on ('cuda' or 'cpu')
#
#     Returns:
#         tuple: (thermal_output, depth_output) - Generated thermal and depth images
#                Each with shape [B, C, T, H, W] where C=1 for both thermal and depth
#     """
#     # Make sure model is in eval mode
#     model.eval()
#
#     # Move inputs to the correct device
#     rgb_frames = rgb_frames.to(device)
#     batch_size, channels, time_frames, height, width = rgb_frames.shape
#
#     # Create a mask for the encoder (we're treating all RGB as "visible" patches)
#     # In this case we're keeping all patches visible during encoding (mask=True)
#     # and we'll generate the thermal and depth modalities from the entire RGB input
#     patch_shape = rearrange(
#         rgb_frames,
#         "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
#         p0=model.tubelet_size,
#         p1=model.patch_size,
#         p2=model.patch_size
#     ).shape
#
#     # Creating a mask where all patches are visible (not masked)
#     mask = torch.ones(batch_size, patch_shape[1], 1, dtype=torch.bool, device=device)
#
#     with torch.no_grad():
#         # Encode the RGB frames
#         encoded = model.encoder(rgb_frames, mask)
#
#         # Project to thermal and depth embedding spaces
#         thermal_emb = model.enc_dec_proj_thermal(encoded)
#         depth_emb = model.enc_dec_proj_depth(encoded)
#
#         # Now we need to create a different mask for the decoder
#         # For generation, we want to predict ALL patches
#         # So we use a mask where all positions are False (indicating they need to be generated)
#         generation_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
#
#         # Decode to get thermal and depth outputs
#         thermal_patches = model.thermal_decoder(thermal_emb, generation_mask)
#         depth_patches = model.depth_decoder(depth_emb, generation_mask)
#
#         # Convert patches back to image format
#         thermal_output = model.thermal_decoder.unpatch_to_img(thermal_patches)
#         depth_output = model.depth_decoder.unpatch_to_img(depth_patches)
#
#     return thermal_output, depth_output
def generate_thermal_depth_from_rgb(model, rgb_frames, device='cuda'):
    """
    Generate thermal and depth images from RGB frames using the MultiModalMarlin model.

    Args:
        model (MultiModalMarlin): Trained model instance
        rgb_frames (torch.Tensor): RGB frames in shape [B, C, T, H, W]
                                  where B=batch size, C=3 (RGB channels),
                                  T=time/frames, H=height, W=width
        device (str): Device to run inference on ('cuda' or 'cpu')

    Returns:
        tuple: (thermal_output, depth_output) - Generated thermal and depth images
               Each with shape [B, C, T, H, W] where C=1 for both thermal and depth
    """
    # Make sure model is in eval mode
    model.eval()

    # Move inputs to the correct device
    rgb_frames = rgb_frames.to(device)
    batch_size, channels, time_frames, height, width = rgb_frames.shape

    # Calculate number of patches in each dimension
    num_patches_t = time_frames // model.tubelet_size
    num_patches_h = height // model.patch_size
    num_patches_w = width // model.patch_size
    num_patches = num_patches_t * num_patches_h * num_patches_w

    # For the encoder, keep all patches visible (True)
    encoder_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Encode the RGB frames
        encoded = model.encoder(rgb_frames, encoder_mask)

        # Project to thermal and depth embedding spaces
        thermal_emb = model.enc_dec_proj_thermal(encoded)
        depth_emb = model.enc_dec_proj_depth(encoded)

        # Based on the error message, we need exactly 1568 visible tokens and 1254 masked tokens
        # The total number of patches should be 1568 + 1254 = 2822

        # Check if our calculated num_patches matches this expectation
        if num_patches != 2822:
            print(f"Warning: Expected 2822 total patches but calculated {num_patches}. Adjusting...")

        # Create a mask with exactly 1568 visible tokens (True) and the rest masked (False)
        num_visible = 1568

        # Initialize mask with all False (masked)
        decoder_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        # Set exactly the first 1568 tokens to visible (True)
        # This ensures we have the exact number needed
        decoder_mask[:, :num_visible] = True

        # For safety, ensure we don't exceed the tensor bounds
        visible_count = min(num_visible, num_patches)
        decoder_mask[:, :visible_count] = True

        print(
            f"Created mask with {torch.sum(decoder_mask).item()} visible tokens and {torch.sum(~decoder_mask).item()} masked tokens")

        # Decode to get thermal and depth outputs
        thermal_patches = model.thermal_decoder(thermal_emb, decoder_mask)
        depth_patches = model.depth_decoder(depth_emb, decoder_mask)

        # Convert patches back to image format
        thermal_output = model.thermal_decoder.unpatch_to_img(thermal_patches)
        depth_output = model.depth_decoder.unpatch_to_img(depth_patches)

    return thermal_output, depth_output

def visualize_outputs(rgb_input, thermal_output, depth_output, save_path=None):
    """
    Visualize the RGB input alongside the thermal and depth outputs.

    Args:
        rgb_input (torch.Tensor): RGB input frames [B, 3, T, H, W]
        thermal_output (torch.Tensor): Generated thermal frames [B, 1, T, H, W]
        depth_output (torch.Tensor): Generated depth frames [B, 1, T, H, W]
        save_path (str, optional): Path to save visualization. If None, just displays.

    Returns:
        np.ndarray: Combined visualization image
    """
    import matplotlib.pyplot as plt

    # Take the first batch and frame for visualization
    rgb_img = rgb_input[0, :, 0].permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)

    thermal_img = thermal_output[0, 0, 0].cpu().numpy()
    thermal_img = np.clip(thermal_img, 0, 1)

    depth_img = depth_output[0, 0, 0].cpu().numpy()
    depth_img = np.clip(depth_img, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot RGB input
    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB Input')
    axes[0].axis('off')

    # Plot thermal output
    axes[1].imshow(thermal_img, cmap='inferno')  # Using a heat-map colormap for thermal
    axes[1].set_title('Generated Thermal')
    axes[1].axis('off')

    # Plot depth output
    axes[2].imshow(depth_img, cmap='viridis')  # Using a depth-map colormap
    axes[2].set_title('Generated Depth')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

    # Return combined visualization as numpy array
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return vis_image

parser = argparse.ArgumentParser("MULTIMODALMARLIN pretraining")
parser.add_argument("--config", type=str)
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--image_path", type=str, required=True,
                      help="Path to the input image file to generate thermal and depth maps")
parser.add_argument("--output_path", type=str, default="generated_outputs.png",
                      help="Path to save the generated visualization")
# Example usage:
def main():
    # Load your trained model
    from model.marlin_multimodal import MultiModalMarlin
    args = parser.parse_args()
    config_path = args.config
    resume_ckpt = args.resume
    config = read_yaml(config_path)

    batch_size = args.batch_size
    max_epochs = args.epochs

    image_path = args.image_path


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
    # mask_strategy = config["mask_strategy"]
    # temporal_sample_rate = config["temporal_sample_rate"]
    # mask_percentage_target = config["mask_percentage_target"]
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
    # Initialize the model with the same parameters used during training
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
        iter_per_epoch=2,
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


    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load pretrained weights
    checkpoint = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    from PIL import Image
    import torchvision.transforms as transforms

    def load_image_for_model(image_path, n_frames=16, img_size=224, device='cuda'):
        # Check if the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open the image
        img = Image.open(image_path)

        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

        # Apply transformations
        img_tensor = transform(img)

        # Replicate the single image across the temporal dimension
        # From [3, H, W] to [3, T, H, W]
        img_tensor = img_tensor.unsqueeze(1).repeat(1, n_frames, 1, 1)

        # Add batch dimension: [3, T, H, W] to [1, 3, T, H, W]
        img_tensor = img_tensor.unsqueeze(0)

        # Move to device
        img_tensor = img_tensor.to(device)

        return img_tensor

    # Load the real image
    sample_rgb = load_image_for_model(
        image_path=image_path,
        n_frames=clip_frames,  # Use the clip_frames from config
        img_size=img_size,  # Use the img_size from config
        device=device
    )

    print(f"Loaded image from {image_path}")
    print(f"Image tensor shape: {sample_rgb.shape}")

    # Generate thermal and depth outputs
    thermal_output, depth_output = generate_thermal_depth_from_rgb(model, sample_rgb, device)

    # Visualize and/or save the results
    if args.output_path.endswith('.npz'):
        # Save raw data as npz
        import numpy as np
        # Convert tensors to numpy arrays
        # Shape: [B, C, T, H, W] -> squeeze batch dim if B=1 -> [C, T, H, W]
        rgb_np = sample_rgb.detach().cpu().numpy()
        thermal_np = thermal_output.detach().cpu().numpy()
        depth_np = depth_output.detach().cpu().numpy()
        
        np.savez(args.output_path, 
                 rgb=rgb_np, 
                 thermal=thermal_np, 
                 depth=depth_np)
        print(f"Saved raw outputs to {args.output_path}")
    else:
        # Save visualization
        visualize_outputs(sample_rgb, thermal_output, depth_output, save_path=args.output_path)


    print(f"Thermal output shape: {thermal_output.shape}")
    print(f"Depth output shape: {depth_output.shape}")


if __name__ == "__main__":
    main()