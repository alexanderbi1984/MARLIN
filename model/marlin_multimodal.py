import itertools
import math
from typing import Optional, Union, Sequence, Tuple, Generator
import cv2
from collections import deque
import ffmpeg
from pathlib import Path
import os.path

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, MSELoss, LeakyReLU
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from model.decoder import MarlinDecoder
from model.encoder import MarlinEncoder
from model.config import resolve_config
from src.marlin_pytorch.face_detector import FaceXZooFaceDetector
# from marlin_pytorch.model.decoder import MarlinDecoder
# from marlin_pytorch.model.encoder import MarlinEncoder
from marlin_pytorch.model.modules import MLP
# from torch.utils.tensorboard import SummaryWriter
from src.marlin_pytorch.util import read_video, padding_video, DownloadProgressBar



class MultiModalMarlin(LightningModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 n_frames=16,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer="LayerNorm",
                 init_values=0.,
                 tubelet_size=2,
                 optimizer_type: str = "AdamW",
                 optimizer_eps: float = 1e-8,
                 optimizer_betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.,
                 learning_rate: float = 1.5e-4,
                 warmup_lr: float = 1e-6,
                 min_lr: float = 1e-5,
                 warmup_epochs: int = 40,
                 max_epochs: int = 2000,
                 iter_per_epoch: int = 1297,
                 distributed: bool = False,
                 d_steps: int = 3,
                 g_steps: int = 1,
                 adv_weight: float = 0.1,
                 gp_weight: float = 10.,
                 rgb_weight: float = 1.0,
                 thermal_weight: float = 1.0,
                 depth_weight: float = 1.0,
                 name: str = None
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Shared encoder for mixed input
        self.encoder = MarlinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )

        # Separate decoders for each modality
        self.rgb_decoder = MarlinDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            output_channels=3
        )

        self.thermal_decoder = MarlinDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            output_channels=1
        )

        self.depth_decoder = MarlinDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            output_channels=1
        )

        # Separate discriminators for each modality
        self.rgb_discriminator = MLP(
            [patch_size * patch_size * 3 * tubelet_size, int(decoder_embed_dim * mlp_ratio), 1],
            build_activation=LeakyReLU)

        self.thermal_discriminator = MLP(
            [patch_size * patch_size * 1 * tubelet_size, int(decoder_embed_dim * mlp_ratio), 1],
            build_activation=LeakyReLU)

        self.depth_discriminator = MLP(
            [patch_size * patch_size * 1 * tubelet_size, int(decoder_embed_dim * mlp_ratio), 1],
            build_activation=LeakyReLU)

        # Projection from encoder to each decoder
        self.enc_dec_proj_rgb = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.enc_dec_proj_thermal = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.enc_dec_proj_depth = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        # Weights for each modality in the loss function
        # print(f"before assignment, the type of rgb_weight is {type(rgb_weight)}")
        self.rgb_weight = rgb_weight
        # print(f"after assignment, the type of self.rgb_weight is {type(self.rgb_weight)}")
        self.thermal_weight = thermal_weight
        self.depth_weight = depth_weight

        # Rest of the initialization is similar to the original
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        # ... original optimizer and scheduler setup code ...

        if optimizer_type == "AdamW":
            self.optimizer_type = AdamW
        elif optimizer_type == "Adam":
            self.optimizer_type = Adam
        else:
            raise ValueError("optimizer_type must be either AdamW or Adam")

        self.optimizer_eps = optimizer_eps
        self.optimizer_betas = optimizer_betas
        self.weight_decay = weight_decay

        self.learning_rate = learning_rate
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch

        self.d_steps = d_steps
        self.g_steps = g_steps
        self.adv_weight = adv_weight
        self.gp_weight = gp_weight

        self.lr_scheduler_factors = self._cosine_scheduler_factors()

        self.loss_fn = MSELoss()
        self.automatic_optimization = False
        self.distributed = distributed
        self.name = name


    def forward(self, x, mask):
        # Encode the mixed input
        encoded = self.encoder(x, mask)

        # Project to different decoder embeddings
        rgb_emb = self.enc_dec_proj_rgb(encoded)
        thermal_emb = self.enc_dec_proj_thermal(encoded)
        depth_emb = self.enc_dec_proj_depth(encoded)

        # Decode each modality
        rgb_out = self.rgb_decoder(rgb_emb, mask)
        thermal_out = self.thermal_decoder(thermal_emb, mask)
        depth_out = self.depth_decoder(depth_emb, mask)

        return rgb_out, thermal_out, depth_out

    @staticmethod 
    def g_loss_fn(pred) -> Tensor:
        return -pred.mean()

    @staticmethod
    def d_loss_fn(pred, target) -> Tensor:
        real_score = pred[target == 1]
        fake_score = pred[target == 0]
        return fake_score.mean() - real_score.mean()

    def gradient_penalty_fn(self, real_patches: Tensor, fake_patches: Tensor) -> Tensor:
        alpha = torch.rand(1).to(self.device).expand(real_patches.size())

        interpolates = torch.autograd.Variable(alpha * real_patches + ((1 - alpha) * fake_patches), requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True)[0]

        gradients = rearrange(gradients, 'b n c -> (b n) c')
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def step(self, batch):
        # Forward step
        mixed_video, mask, rgb_frames, depth_frames, thermal_frames = batch

        # Get predictions for each modality
        rgb_pred, thermal_pred, depth_pred = self(mixed_video, mask)

        # Get patches for ground truth modalities
        # RGB patches
        rgb_patches = rgb_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        rgb_patches = rearrange(rgb_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Thermal patches
        thermal_patches = thermal_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        thermal_patches = rearrange(thermal_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Depth patches
        depth_patches = depth_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        depth_patches = rearrange(depth_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Filter masked patches for each modality
        b, _, c_rgb = rgb_patches.shape
        b, _, c_thermal = thermal_patches.shape
        b, _, c_depth = depth_patches.shape

        rgb_patches_masked = rgb_patches[~mask].view(b, -1, c_rgb)
        thermal_patches_masked = thermal_patches[~mask].view(b, -1, c_thermal)
        depth_patches_masked = depth_patches[~mask].view(b, -1, c_depth)

        return (rgb_pred, thermal_pred, depth_pred), (rgb_patches_masked, thermal_patches_masked, depth_patches_masked)


    # def g_step(self, preds, targets):
    #     rgb_pred, thermal_pred, depth_pred = preds
    #     rgb_target, thermal_target, depth_target = targets
    #
    #     # Reconstruction losses
    #     rgb_rec_loss = self.loss_fn(rgb_pred, rgb_target)
    #     thermal_rec_loss = self.loss_fn(thermal_pred, thermal_target)
    #     depth_rec_loss = self.loss_fn(depth_pred, depth_target)
    #
    #     # Combined reconstruction loss with weights
    #     rec_loss = (
    #             self.rgb_weight * rgb_rec_loss +
    #             self.thermal_weight * thermal_rec_loss +
    #             self.depth_weight * depth_rec_loss
    #     )
    #
    #     # Adversarial losses
    #     rgb_adv_loss = self.adv_weight * self.g_loss_fn(self.rgb_discriminator(rgb_pred)).mean()
    #     thermal_adv_loss = self.adv_weight * self.g_loss_fn(self.thermal_discriminator(thermal_pred)).mean()
    #     depth_adv_loss = self.adv_weight * self.g_loss_fn(self.depth_discriminator(depth_pred)).mean()
    #
    #     # Combined adversarial loss
    #     adv_loss = rgb_adv_loss + thermal_adv_loss + depth_adv_loss
    #
    #     # Total loss
    #     total_loss = rec_loss + adv_loss
    #
    #     return {
    #         "loss": total_loss,
    #         "g_loss": total_loss,
    #         "rgb_rec_loss": rgb_rec_loss,
    #         "thermal_rec_loss": thermal_rec_loss,
    #         "depth_rec_loss": depth_rec_loss,
    #         "rgb_adv_loss": rgb_adv_loss,
    #         "thermal_adv_loss": thermal_adv_loss,
    #         "depth_adv_loss": depth_adv_loss
    #     }

    def g_step(self, preds, targets):
        rgb_pred, thermal_pred, depth_pred = preds
        rgb_target, thermal_target, depth_target = targets

        # Reconstruction losses
        rgb_rec_loss = self.loss_fn(rgb_pred, rgb_target)
        thermal_rec_loss = self.loss_fn(thermal_pred, thermal_target)
        depth_rec_loss = self.loss_fn(depth_pred, depth_target)

        # Convert weights to tensors if they aren't already
        # print(f"the type of the rgb_weight:{type(self.rgb_weight)}")
        rgb_weight = torch.tensor(self.rgb_weight, device=rgb_rec_loss.device, dtype=rgb_rec_loss.dtype)
        thermal_weight = torch.tensor(self.thermal_weight, device=thermal_rec_loss.device, dtype=thermal_rec_loss.dtype)
        depth_weight = torch.tensor(self.depth_weight, device=depth_rec_loss.device, dtype=depth_rec_loss.dtype)

        # Combined reconstruction loss with weights
        rec_loss = (
                rgb_weight * rgb_rec_loss +
                thermal_weight * thermal_rec_loss +
                depth_weight * depth_rec_loss
        )

        # Adversarial losses - ensure we're not using tensors as indices
        adv_weight = torch.tensor(self.adv_weight, device=rgb_pred.device, dtype=rgb_pred.dtype)

        # Compute discriminator outputs
        rgb_disc_out = self.rgb_discriminator(rgb_pred)
        thermal_disc_out = self.thermal_discriminator(thermal_pred)
        depth_disc_out = self.depth_discriminator(depth_pred)

        # Apply g_loss_fn
        rgb_adv_loss = adv_weight * self.g_loss_fn(rgb_disc_out).mean()
        thermal_adv_loss = adv_weight * self.g_loss_fn(thermal_disc_out).mean()
        depth_adv_loss = adv_weight * self.g_loss_fn(depth_disc_out).mean()

        # Combined adversarial loss
        adv_loss = rgb_adv_loss + thermal_adv_loss + depth_adv_loss

        # Total loss
        total_loss = rec_loss + adv_loss

        return {
            "loss": total_loss,
            "g_loss": total_loss,
            "rgb_rec_loss": rgb_rec_loss,
            "thermal_rec_loss": thermal_rec_loss,
            "depth_rec_loss": depth_rec_loss,
            "rgb_adv_loss": rgb_adv_loss,
            "thermal_adv_loss": thermal_adv_loss,
            "depth_adv_loss": depth_adv_loss
        }


    def d_step(self, preds, targets):
        rgb_pred, thermal_pred, depth_pred = preds
        rgb_target, thermal_target, depth_target = targets

        # RGB discriminator loss
        rgb_fake_labels = torch.zeros(rgb_pred.size(0), rgb_pred.size(1), 1, device=self.device)
        rgb_real_labels = torch.ones(rgb_target.size(0), rgb_target.size(1), 1, device=self.device)
        rgb_d_batch = torch.cat((rgb_pred, rgb_target), dim=0)
        rgb_d_labels = torch.cat((rgb_fake_labels, rgb_real_labels), dim=0)
        rgb_d_loss = self.d_loss_fn(self.rgb_discriminator(rgb_d_batch), rgb_d_labels).mean()

        # Thermal discriminator loss
        thermal_fake_labels = torch.zeros(thermal_pred.size(0), thermal_pred.size(1), 1, device=self.device)
        thermal_real_labels = torch.ones(thermal_target.size(0), thermal_target.size(1), 1, device=self.device)
        thermal_d_batch = torch.cat((thermal_pred, thermal_target), dim=0)
        thermal_d_labels = torch.cat((thermal_fake_labels, thermal_real_labels), dim=0)
        thermal_d_loss = self.d_loss_fn(self.thermal_discriminator(thermal_d_batch), thermal_d_labels).mean()

        # Depth discriminator loss
        depth_fake_labels = torch.zeros(depth_pred.size(0), depth_pred.size(1), 1, device=self.device)
        depth_real_labels = torch.ones(depth_target.size(0), depth_target.size(1), 1, device=self.device)
        depth_d_batch = torch.cat((depth_pred, depth_target), dim=0)
        depth_d_labels = torch.cat((depth_fake_labels, depth_real_labels), dim=0)
        depth_d_loss = self.d_loss_fn(self.depth_discriminator(depth_d_batch), depth_d_labels).mean()

        # Combined discriminator loss
        d_loss = rgb_d_loss + thermal_d_loss + depth_d_loss

        # Gradient penalty if needed
        if self.training and self.gp_weight > 0:
            rgb_gp = self.gradient_penalty_fn(rgb_target, rgb_pred.detach())
            thermal_gp = self.gradient_penalty_fn(thermal_target, thermal_pred.detach())
            depth_gp = self.gradient_penalty_fn(depth_target, depth_pred.detach())

            gp = rgb_gp + thermal_gp + depth_gp
            total_loss = d_loss + self.gp_weight * gp

            return {
                "loss": total_loss,
                "d_loss": total_loss,
                "d_loss0": d_loss,
                "rgb_d_loss": rgb_d_loss,
                "thermal_d_loss": thermal_d_loss,
                "depth_d_loss": depth_d_loss,
                "gp": gp
            }
        else:
            return {
                "loss": d_loss,
                "d_loss": d_loss,
                "d_loss0": d_loss,
                "rgb_d_loss": rgb_d_loss,
                "thermal_d_loss": thermal_d_loss,
                "depth_d_loss": depth_d_loss
            }

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
    ) -> Tensor:
        g_optimizer, d_optimizer = self.optimizers()
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            g_scheduler, d_scheduler = schedulers
        else:
            g_scheduler, d_scheduler = None, None

        # Train discriminator
        d_loss = None
        d_result = None
        for _ in range(self.d_steps):
            d_optimizer.zero_grad()
            preds, targets = self.step(batch)
            d_result = self.d_step(preds, targets)
            d_loss = d_result["loss"]
            self.manual_backward(d_loss)
            d_optimizer.step()
        if d_scheduler is not None and batch_idx == 0:
            d_scheduler.step()

        # Train generator (encoders and decoders)
        g_loss = None
        g_result = None
        for _ in range(self.g_steps):
            g_optimizer.zero_grad()
            preds, targets = self.step(batch)
            g_result = self.g_step(preds, targets)
            g_loss = g_result["loss"]
            self.manual_backward(g_loss)
            g_optimizer.step()
        if g_scheduler is not None and batch_idx == 0:
            g_scheduler.step()

        loss_dict = {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                        dataloader_idx: Optional[int] = None
                        ) -> Tensor:
        # Log sample reconstruction image at the beginning of each validation epoch
        if batch_idx == 0:
            self._log_sample_reconstruction_image(batch)

        # Unpack the batch
        mixed_video, mask, rgb_frames, depth_frames, thermal_frames = batch

        # Get predictions for each modality
        rgb_pred, thermal_pred, depth_pred = self(mixed_video, mask)

        # Process ground truth targets for each modality
        # RGB patches
        rgb_patches = rgb_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        rgb_patches = rearrange(rgb_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Thermal patches
        thermal_patches = thermal_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        thermal_patches = rearrange(thermal_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Depth patches
        depth_patches = depth_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        depth_patches = rearrange(depth_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Filter masked patches
        b, _, c_rgb = rgb_patches.shape
        b, _, c_thermal = thermal_patches.shape
        b, _, c_depth = depth_patches.shape

        rgb_patches_masked = rgb_patches[~mask].view(b, -1, c_rgb)
        thermal_patches_masked = thermal_patches[~mask].view(b, -1, c_thermal)
        depth_patches_masked = depth_patches[~mask].view(b, -1, c_depth)

        # Prepare inputs for loss calculation
        preds = (rgb_pred, thermal_pred, depth_pred)
        targets = (rgb_patches_masked, thermal_patches_masked, depth_patches_masked)

        # Calculate discriminator loss
        d_result = self.d_step(preds, targets)
        d_loss = d_result["loss"]

        # Calculate generator loss
        g_result = self.g_step(preds, targets)
        g_loss = g_result["loss"]

        # Combine all losses and metrics
        loss_dict = {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }

        # Log all metrics
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
                      prog_bar=True, sync_dist=self.distributed)

        return loss_dict["loss"]


    def _cosine_scheduler_factors(self):
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * self.iter_per_epoch
        if self.warmup_epochs > 0:
            warmup_schedule = np.linspace(0, self.learning_rate, warmup_iters)

        iters = np.arange(self.max_epochs * self.iter_per_epoch - warmup_iters)
        schedule = np.array(
            [self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * i / (len(iters))))
                for i in iters])

        schedule = np.concatenate((warmup_schedule, schedule))

        assert len(schedule) == self.max_epochs * self.iter_per_epoch
        values_factors = schedule[::self.iter_per_epoch] / self.learning_rate
        return values_factors


    def _cosine_scheduler_fn(self, epoch):
        return self.lr_scheduler_factors[epoch]


    def configure_optimizers(self):
        g_optimizer = self.optimizer_type(
            itertools.chain(
                self.encoder.parameters(),
                self.rgb_decoder.parameters(),
                self.thermal_decoder.parameters(),
                self.depth_decoder.parameters(),
                self.enc_dec_proj_rgb.parameters(),
                self.enc_dec_proj_thermal.parameters(),
                self.enc_dec_proj_depth.parameters()
            ),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay
        )

        g_lr_scheduler = LambdaLR(
            g_optimizer,
            lr_lambda=self._cosine_scheduler_fn
        )

        d_optimizer = self.optimizer_type(
            itertools.chain(
                self.rgb_discriminator.parameters(),
                self.thermal_discriminator.parameters(),
                self.depth_discriminator.parameters()
            ),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay
        )

        d_lr_scheduler = LambdaLR(
            d_optimizer,
            lr_lambda=self._cosine_scheduler_fn
        )

        return [g_optimizer, d_optimizer], [g_lr_scheduler, d_lr_scheduler]


    # def _log_sample_reconstruction_image(self, batch):
    #     mixed_video, mask, rgb_frames, depth_frames, thermal_frames = batch
    #     # Print shapes to diagnose the issue
    #     print(f"Mixed video shape: {mixed_video.shape}")
    #     print(f"Mask shape: {mask.shape}")
    #     print(f"RGB frames shape: {rgb_frames.shape}")
    #     # Take only the first batch item for visualization
    #     mixed_video = mixed_video[:1]
    #     mask = mask[:1]
    #     rgb_frames = rgb_frames[:1]
    #     depth_frames = depth_frames[:1]
    #     thermal_frames = thermal_frames[:1]
    #
    #     # Get predictions
    #     rgb_pred, thermal_pred, depth_pred = self(mixed_video, mask)
    #
    #     # Print more shapes
    #     print(f"RGB prediction shape: {rgb_pred.shape}")
    #     print(
    #         f"Expected number of patches: {mixed_video.shape[2] // self.tubelet_size * (mixed_video.shape[3] // self.patch_size) * (mixed_video.shape[4] // self.patch_size)}")
    #
    #     # Prepare RGB visualization
    #     rgb_gt_img = rgb_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
    #         .unfold(3, self.patch_size, self.patch_size) \
    #         .unfold(4, self.patch_size, self.patch_size)
    #     rgb_gt_img = rearrange(rgb_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
    #     rgb_gt_img = self.rgb_decoder.unpatch_to_img(rgb_gt_img).detach()[0, :, 0]  # (C, H, W)
    #
    #     # Prepare Thermal visualization
    #     thermal_gt_img = thermal_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
    #         .unfold(3, self.patch_size, self.patch_size) \
    #         .unfold(4, self.patch_size, self.patch_size)
    #     thermal_gt_img = rearrange(thermal_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
    #     thermal_gt_img = self.thermal_decoder.unpatch_to_img(thermal_gt_img).detach()[0, :, 0]  # (C, H, W)
    #
    #     # Prepare Depth visualization
    #     depth_gt_img = depth_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
    #         .unfold(3, self.patch_size, self.patch_size) \
    #         .unfold(4, self.patch_size, self.patch_size)
    #     depth_gt_img = rearrange(depth_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
    #     depth_gt_img = self.depth_decoder.unpatch_to_img(depth_gt_img).detach()[0, :, 0]  # (C, H, W)
    #
    #     # Make patched versions for reconstruction
    #     # Patch mixed input
    #     x = rearrange(mixed_video, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
    #                   p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
    #     x = rearrange(x, "b n p c -> b n (c p)")
    #
    #     # Make masked visualization
    #     x_masked = x.clone()
    #     x_masked[~mask] = 0.5
    #     masked_img = self.rgb_decoder.unpatch_to_img(x_masked).detach()[0, :, 0]  # Using RGB decoder for visualization
    #
    #     # Reconstructed images
    #     rgb_rec_img = self.rgb_decoder.unpatch_to_img(rgb_pred).detach()[0, :, 0]
    #     thermal_rec_img = self.thermal_decoder.unpatch_to_img(thermal_pred).detach()[0, :, 0]
    #     depth_rec_img = self.depth_decoder.unpatch_to_img(depth_pred).detach()[0, :, 0]
    #
    #     # Create visualization grid
    #     # RGB row
    #     rgb_row = torch.cat([rgb_gt_img, masked_img, rgb_rec_img], dim=2)
    #
    #     # Thermal row (repeat channels to make it visible if it's single-channel)
    #     if thermal_gt_img.size(0) == 1:
    #         thermal_gt_img = thermal_gt_img.repeat(3, 1, 1)
    #         thermal_rec_img = thermal_rec_img.repeat(3, 1, 1)
    #     thermal_row = torch.cat([thermal_gt_img, masked_img, thermal_rec_img], dim=2)
    #
    #     # Depth row (repeat channels to make it visible if it's single-channel)
    #     if depth_gt_img.size(0) == 1:
    #         depth_gt_img = depth_gt_img.repeat(3, 1, 1)
    #         depth_rec_img = depth_rec_img.repeat(3, 1, 1)
    #     depth_row = torch.cat([depth_gt_img, masked_img, depth_rec_img], dim=2)
    #
    #     # Final grid
    #     log_img = torch.cat([rgb_row, thermal_row, depth_row], dim=1)
    #     self.log_image("sample_multimodal", log_img)
    def _log_sample_reconstruction_image(self, batch):
        mixed_video, mask, rgb_frames, depth_frames, thermal_frames = batch
        # Print shapes to diagnose the issue
        # print(f"Mixed video shape: {mixed_video.shape}")
        # print(f"Mask shape: {mask.shape}")
        # print(f"RGB frames shape: {rgb_frames.shape}")

        # Take only the first batch item for visualization
        mixed_video = mixed_video[:1]
        mask = mask[:1]
        rgb_frames = rgb_frames[:1]
        depth_frames = depth_frames[:1]
        thermal_frames = thermal_frames[:1]

        # Get predictions (these will be just the masked patches)
        rgb_pred, thermal_pred, depth_pred = self(mixed_video, mask)

        # Print more shapes
        # print(f"RGB prediction shape: {rgb_pred.shape}")
        expected_patches = mixed_video.shape[2] // self.tubelet_size * (mixed_video.shape[3] // self.patch_size) * (
                    mixed_video.shape[4] // self.patch_size)
        # print(f"Expected number of patches: {expected_patches}")

        # Count visible and masked patches
        # visible_patches = mask.sum().item()
        # masked_patches = mask.numel() - visible_patches
        # print(f"Visible patches: {visible_patches}, Masked patches: {masked_patches}")

        # Create full tensors that include all patches for visualization
        # b = mixed_video.shape[0]  # Should be 1 for visualization

        # # Create empty tensors for visualization with all patches
        # rgb_full = torch.zeros(b, expected_patches, rgb_pred.shape[-1], device=rgb_pred.device)
        # thermal_full = torch.zeros(b, expected_patches, thermal_pred.shape[-1], device=thermal_pred.device)
        # depth_full = torch.zeros(b, expected_patches, depth_pred.shape[-1], device=depth_pred.device)
        #
        # # Fill in the reconstructed patches only at the masked positions
        # # The decoder outputs are in the same order as the masked positions
        # rgb_full[~mask] = rgb_pred
        # thermal_full[~mask] = thermal_pred
        # depth_full[~mask] = depth_pred

        # Create full tensors with original content first
        rgb_patches = rgb_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        rgb_patches = rearrange(rgb_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        thermal_patches = thermal_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        thermal_patches = rearrange(thermal_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        depth_patches = depth_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        depth_patches = rearrange(depth_patches, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")

        # Start with the original content for each modality
        rgb_full = rgb_patches.clone()
        thermal_full = thermal_patches.clone()
        depth_full = depth_patches.clone()

        # Replace only the masked regions with predictions
        rgb_full[~mask] = rgb_pred
        thermal_full[~mask] = thermal_pred
        depth_full[~mask] = depth_pred

        # Prepare RGB visualization
        rgb_gt_img = rgb_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        rgb_gt_img = rearrange(rgb_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        rgb_gt_img = self.rgb_decoder.unpatch_to_img(rgb_gt_img).detach()[0, :, 0]  # (C, H, W)

        # Prepare Thermal visualization
        thermal_gt_img = thermal_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        thermal_gt_img = rearrange(thermal_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        thermal_gt_img = self.thermal_decoder.unpatch_to_img(thermal_gt_img).detach()[0, :, 0]  # (C, H, W)

        # Prepare Depth visualization
        depth_gt_img = depth_frames.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        depth_gt_img = rearrange(depth_gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        depth_gt_img = self.depth_decoder.unpatch_to_img(depth_gt_img).detach()[0, :, 0]  # (C, H, W)

        # Make patched versions for reconstruction
        # Patch mixed input
        x = rearrange(mixed_video, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                      p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, "b n p c -> b n (c p)")

        # Make masked visualization
        x_masked = x.clone()
        x_masked[~mask] = 0.5
        masked_img = self.rgb_decoder.unpatch_to_img(x_masked).detach()[0, :, 0]  # Using RGB decoder for visualization

        try:
            # Try to visualize using the full tensors we created
            rgb_rec_img = self.rgb_decoder.unpatch_to_img(rgb_full).detach()[0, :, 0]
            thermal_rec_img = self.thermal_decoder.unpatch_to_img(thermal_full).detach()[0, :, 0]
            depth_rec_img = self.depth_decoder.unpatch_to_img(depth_full).detach()[0, :, 0]
        except Exception as e:
            print(f"Visualization error: {e}")
            # Fallback: create simplified visualization
            # Just show a grid of the reconstructed patches
            print("Using fallback visualization method")

            # Create a blank canvas for each modality
            rgb_rec_img = torch.zeros_like(rgb_gt_img)
            thermal_rec_img = torch.zeros_like(thermal_gt_img)
            depth_rec_img = torch.zeros_like(depth_gt_img)

            # Place text or indicator that visualization failed
            # (in a real implementation, you might want a more sophisticated fallback)

        # Create visualization grid
        # RGB row
        rgb_row = torch.cat([rgb_gt_img, masked_img, rgb_rec_img], dim=2)

        # Thermal row (repeat channels to make it visible if it's single-channel)
        if thermal_gt_img.size(0) == 1:
            thermal_gt_img = thermal_gt_img.repeat(3, 1, 1)
            thermal_rec_img = thermal_rec_img.repeat(3, 1, 1)
        thermal_row = torch.cat([thermal_gt_img, masked_img, thermal_rec_img], dim=2)

        # Depth row (repeat channels to make it visible if it's single-channel)
        if depth_gt_img.size(0) == 1:
            depth_gt_img = depth_gt_img.repeat(3, 1, 1)
            depth_rec_img = depth_rec_img.repeat(3, 1, 1)
        depth_row = torch.cat([depth_gt_img, masked_img, depth_rec_img], dim=2)

        # Final grid
        log_img = torch.cat([rgb_row, thermal_row, depth_row], dim=1)
        self.log_image("sample_multimodal", log_img)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """Log an image to the logger"""
        if self.logger is None:
            return
        self.logger.experiment.add_image(name, image, self.current_epoch)


    @classmethod
    def from_file(cls, model_name: str, path: str) -> "MultiModalMarlin":
        """
        Load a MultiModalMarlin model from a checkpoint file.

        Args:
            model_name (str): The name of the model configuration to use.
            path (str): Path to the checkpoint file (.pt or .ckpt).

        Returns:
            MultiModalMarlin: The loaded model.

        Raises:
            ValueError: If the file type is not supported.
        """
        if path.endswith(".pt"):
            state_dict = torch.load(path, map_location="cpu")
        elif path.endswith(".ckpt"):
            state_dict = torch.load(path, map_location="cpu")["state_dict"]

            # Remove discriminator keys if present (optional)
            discriminator_keys = [k for k in state_dict.keys()
                                  if k.startswith("rgb_discriminator") or
                                  k.startswith("thermal_discriminator") or
                                  k.startswith("depth_discriminator")]
            for key in discriminator_keys:
                del state_dict[key]
        else:
            raise ValueError(f"Unsupported file type: {path.split('.')[-1]}")

        # Determine if the checkpoint is encoder only by checking for decoder keys
        as_feature_extractor = True
        decoder_prefixes = ["rgb_decoder.", "thermal_decoder.", "depth_decoder."]
        for prefix in decoder_prefixes:
            if any(key.startswith(prefix) for key in state_dict.keys()):
                as_feature_extractor = False
                break

        config = resolve_config(model_name)
        model = cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=config.n_frames,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
            # Additional parameters specific to MultiModalMarlin
            distributed=False,  # Default values
            d_steps=config.get("d_steps", 1),  # Default to 1 as per your config
            g_steps=config.get("g_steps", 1),
            adv_weight=config.get("adv_weight", 0.01),  # Default to 0.01 as per your config
            gp_weight=config.get("gp_weight", 0.0),  # Default to 0.0 as per your config
            rgb_weight=config.get("rgb_weight", 1.0),
            thermal_weight=config.get("thermal_weight", 1.0),
            depth_weight=config.get("depth_weight", 1.0),
            name=model_name
        )

        # Load the state dictionary
        # model.load_state_dict(state_dict)
        encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in encoder_state_dict.items()}
        model.encoder.load_state_dict(encoder_state_dict)
        return model

    @torch.no_grad()
    def extract_video(self, video_path: str, modality: str = "rgb", crop_face: bool = False, sample_rate: int = 2,
                      stride: int = 16, reduction: str = "none", keep_seq: bool = False,
                      detector_device: Optional[str] = None) -> Tensor:
        """
        Extract features from a video using the MultiModalMarlin encoder.

        Args:
            video_path (str): Path to the video file.
            modality (str): Which modality to process the video as. Options: "rgb", "thermal", "depth", or "mixed".
                If "mixed", the video will be treated as mixed multimodal input.
            crop_face (bool): Whether to crop faces from the video frames.
            sample_rate (int): Rate at which to sample frames from the video.
            stride (int): Stride size for sliding window when processing long videos.
            reduction (str): How to reduce features across frames. Options: "none", "mean", "max".
            keep_seq (bool): Whether to keep the sequence dimension in the output features.
            detector_device (Optional[str]): Device to use for face detection.

        Returns:
            Tensor: Extracted features.
        """
        self.eval()
        features = []

        for v in self._load_video(video_path, sample_rate, stride):
            # v: (1, C, T, H, W)
            if crop_face:
                if not FaceXZooFaceDetector.inited:
                    Path(".marlin").mkdir(exist_ok=True)
                    FaceXZooFaceDetector.init(
                        face_sdk_path=FaceXZooFaceDetector.install(os.path.join(".marlin", "FaceXZoo")),
                        device=detector_device or self.device
                    )
                v = self._crop_face(v)

            assert v.shape[3:] == (224, 224), f"Expected shape (224, 224), got {v.shape[3:]}"

            # Create dummy mask for encoder (all True = visible)
            batch_size = v.shape[0]
            num_patches = (self.encoder.pos_embedding.input_shape[0] // batch_size)
            mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=v.device)

            # Process according to modality
            if modality == "mixed":
                # Already mixed input
                encoded = self.encoder(v, mask)
            else:
                # Convert to specified modality (project to all three, then extract the one we want)
                encoded = self.encoder(v, mask)

                if modality == "rgb":
                    encoded = self.enc_dec_proj_rgb(encoded)
                elif modality == "thermal":
                    encoded = self.enc_dec_proj_thermal(encoded)
                elif modality == "depth":
                    encoded = self.enc_dec_proj_depth(encoded)
                else:
                    raise ValueError(f"Unknown modality: {modality}. "
                                     f"Options are 'rgb', 'thermal', 'depth', or 'mixed'.")

            # Handle sequence pooling
            if not keep_seq:
                encoded = encoded.mean(dim=1)  # Pool over sequence dimension

            features.append(encoded)

        features = torch.cat(features)  # Concatenate all features

        if reduction == "mean":
            return features.mean(dim=0)
        elif reduction == "max":
            return features.max(dim=0)[0]

        return features

    def _load_video(self, video_path: str, sample_rate: int, stride: int) -> Generator[Tensor, None, None]:
        """
        Load and preprocess video frames.

        Args:
            video_path (str): Path to the video file.
            sample_rate (int): Sample rate for frame extraction.
            stride (int): Stride for sliding window.

        Yields:
            Tensor: Processed video clips.
        """
        probe = ffmpeg.probe(video_path)
        total_frames = int(probe["streams"][0]["nb_frames"])

        if total_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # Pad frames to match clip_frames
            v = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            assert v.shape[0] == self.clip_frames
            yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        elif total_frames <= self.clip_frames * sample_rate:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # Use first clip_frames frames
            if video.shape[0] < self.clip_frames:
                # Double-check the number of frames
                v = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            else:
                v = video[:self.clip_frames]
            yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        else:
            # Extract features based on sliding window
            cap = cv2.VideoCapture(video_path)
            deq = deque(maxlen=self.clip_frames)

            clip_start_indexes = list(range(0, total_frames - self.clip_frames * sample_rate, stride * sample_rate))
            clip_end_indexes = [i + self.clip_frames * sample_rate - 1 for i in clip_start_indexes]

            current_index = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_index += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1) / 255  # (C, H, W)

                for _ in range(sample_rate - 1):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    current_index += 1

                deq.append(frame)
                if current_index in clip_end_indexes:
                    v = torch.stack(list(deq))  # (T, C, H, W)
                    yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

            cap.release()

    def _crop_face(self, v: Tensor) -> Tensor:
        """
        Crop faces from video frames.

        Args:
            v (Tensor): Video tensor of shape (1, C, T, H, W)

        Returns:
            Tensor: Face-cropped video tensor.
        """
        # Use face SDK to crop face
        v = (rearrange(v, "b c t h w -> (b t) h w c").cpu().numpy() * 255).astype(np.uint8)
        face_frames = []
        for i in range(v.shape[0]):
            # crop_face result: (H, W, C)
            face_results = FaceXZooFaceDetector.crop_face(v[i])
            if len(face_results) > 0:
                face_frames.append(torch.from_numpy(face_results[0]))
            else:
                # If no face detected, use original frame
                face_frames.append(torch.from_numpy(v[i]))

        faces = torch.stack(face_frames)  # (T, H, W, C)
        return rearrange(faces, "(b t) h w c -> b c t h w", b=1).to(self.device) / 255