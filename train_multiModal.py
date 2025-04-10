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


    # if official_pretrained is not None:
    #     print(f"loading official pretrained model from {official_pretrained}")
    #     print(load_official_pretrain_model(model, official_pretrained))
    if official_pretrained is not None:
        # Check if the file exists
        if not os.path.exists(official_pretrained):
            raise FileNotFoundError(f"The specified checkpoint file does not exist: {official_pretrained}")

        print(f"Loading official pretrained model from {official_pretrained}")
        try:
            load_result = load_official_pretrain_model(model, official_pretrained)
            print("Successfully loaded model with the following result:", load_result)
        except Exception as e:
            print(f"Failed to load the official pretrained model: {e}")
    # ddp is not available on windows. need to use dp
    # strategy = "auto" if n_gpus <= 1 else "ddp"
    # # accelerator = "cpu" if n_gpus == 0 else "gpu"
    # accelerator = strategy
    # # accelerator = None if n_gpus <= 1 else "ddp"
    # device = "gpu" if n_gpus > 0 else "cpu"
    # n_gpus = n_gpus if n_gpus > 0 else None
    # print(f"checkpointing to ckpt/{resume_ckpt}")
    # # trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=device,
    # #     logger=True, precision=32, max_epochs=max_epochs,
    # #     strategy=accelerator, resume_from_checkpoint=resume_ckpt,
    # #     callbacks=[ModelCheckpoint(dirpath=f"ckpt/{model_name}", save_last=True,
    # #         filename=model.name + "-{epoch}-{val_loss:.3f}",
    # #         monitor="val_loss", mode="min")])
    # trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=device,
    #                   logger=True, precision=32, max_epochs=max_epochs,
    #                   strategy=accelerator,
    #                   callbacks=[ModelCheckpoint(dirpath=f"ckpt/{model_name}", save_last=True,
    #                                              filename=model.name + "-{epoch}-{val_loss:.3f}",
    #                                              monitor="val_loss", mode="min")])




    # def configure_trainer(n_gpus, max_epochs, model_name):
    #     # Check the operating system
    #     is_windows = platform.system() == "Windows"
    #
    #     # Device configuration (same for both platforms)
    #     accelerator = "gpu" if n_gpus > 0 else "cpu"
    #     devices = n_gpus if n_gpus > 0 else None
    #
    #     # Strategy configuration - differs between platforms
    #     if n_gpus <= 1:
    #         # For single GPU or CPU, use "auto" strategy on both platforms
    #         strategy = "auto"
    #     else:
    #         # For multiple GPUs, use different strategies based on platform
    #         if is_windows:
    #             strategy = "auto"
    #             print("Using auto strategy for multi-GPU on Windows")
    #         else:
    #             strategy = "ddp"  # Distributed Data Parallel works on Linux
    #             print("Using Distributed Data Parallel strategy for multi-GPU on Linux")
    #
    #     # Checkpoint callback (same for both platforms)
    #     checkpoint_callback = ModelCheckpoint(
    #         dirpath=f"ckpt/{model_name}",
    #         save_last=True,
    #         filename=model.name + "-{epoch}-{val_loss:.3f}",
    #         monitor="val_loss",
    #         mode="min"
    #     )
    #
    #     # Create and return trainer
    #     trainer = Trainer(
    #         log_every_n_steps=1,
    #         devices=devices,
    #         accelerator=accelerator,
    #         logger=True,
    #         precision=32,
    #         max_epochs=max_epochs,
    #         strategy=strategy,
    #         callbacks=[checkpoint_callback]
    #     )
    #
    #     print(f"Trainer configured with: accelerator={accelerator}, devices={devices}, strategy={strategy}")
    #     return trainer

    def configure_trainer(n_gpus, max_epochs, model_name):
        # Check the operating system
        is_windows = platform.system() == "Windows"

        # Device configuration (same for both platforms)
        accelerator = "gpu" if n_gpus > 0 else "cpu"
        devices = n_gpus if n_gpus > 0 else None

        # Strategy configuration - differs between platforms
        if n_gpus <= 1:
            # For single GPU or CPU, use "auto" strategy on both platforms
            strategy = "auto"
        else:
            # For multiple GPUs, use different strategies based on platform
            if is_windows:
                # Force 'gloo' backend for Windows multi-GPU setup
                # Create an explicit DDPStrategy with gloo backend
                from pytorch_lightning.strategies import DDPStrategy
                strategy = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)
                print("Using Distributed Data Parallel strategy with gloo backend for multi-GPU on Windows")
            else:
                strategy = "ddp"  # Distributed Data Parallel works on Linux with NCCL by default
                print("Using Distributed Data Parallel strategy for multi-GPU on Linux")

        # Checkpoint callback (same for both platforms)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"ckpt/{model_name}",
            save_last=True,
            filename=model.name + "-{epoch}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min"
        )

        # Create and return trainer
        trainer = Trainer(
            log_every_n_steps=1,
            devices=devices,
            accelerator=accelerator,
            logger=True,
            precision=32,
            max_epochs=max_epochs,
            strategy=strategy,
            callbacks=[checkpoint_callback]
        )

        print(f"Trainer configured with: accelerator={accelerator}, devices={devices}, strategy={strategy}")
        return trainer


    trainer = configure_trainer(
        n_gpus=n_gpus,
        max_epochs=max_epochs,
        model_name="multimodal_marlin",
    )
    if resume_ckpt:  # Check if a checkpoint path is provided
        trainer.fit(model, dm, ckpt_path=resume_ckpt)  # Resume training from the checkpoint
    else:
        trainer.fit(model, dm)  # Start training from scratch
    # trainer.fit(model, dm)
