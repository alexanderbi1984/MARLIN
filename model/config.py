from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Dict, Any

from marlin_pytorch.util import read_yaml, Singleton, NoArgInit


@dataclass
class MarlinConfig:
    img_size: int
    patch_size: int
    n_frames: int
    encoder_embed_dim: int
    encoder_depth: int
    encoder_num_heads: int
    decoder_embed_dim: int
    decoder_depth: int
    decoder_num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: Optional[float]
    drop_rate: float
    attn_drop_rate: float
    norm_layer: str
    init_values: float
    tubelet_size: int

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with a default.
        This allows accessing config attributes that might not exist in all configs.

        Args:
            key: The attribute name to look for
            default: The default value to return if the attribute doesn't exist

        Returns:
            The attribute value or the default
        """
        return getattr(self, key, default)


@dataclass
class MultiModalMarlinConfig(MarlinConfig):
    """Configuration class for MultiModalMarlin model"""
    # Additional parameters specific to MultiModalMarlin
    d_steps: int = 3
    g_steps: int = 1
    adv_weight: float = 0.1
    gp_weight: float = 10.0
    rgb_weight: float = 1.0
    thermal_weight: float = 1.0
    depth_weight: float = 1.0


class Downloadable(ABC):

    @property
    @abstractmethod
    def full_model_url(self) -> str:
        pass

    @property
    @abstractmethod
    def encoder_model_url(self) -> str:
        pass


T = TypeVar("T", bound=MarlinConfig)

_configs = {}


def register_model(name: str):
    def wrapper(cls: Type[T]):
        _configs[name] = cls
        return cls

    return wrapper


class SharedConfig(MarlinConfig):
    img_size = 224
    patch_size = 16
    n_frames = 16
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    norm_layer = "LayerNorm"
    init_values = 0.
    tubelet_size = 2


class SharedMultiModalConfig(MultiModalMarlinConfig, SharedConfig):
    """Base configuration for MultiModal models with common settings"""
    d_steps = 1  # Changed from 3 to 1 as per your config
    g_steps = 1
    adv_weight = 0.01  # Changed from 0.1 to 0.01 as per your config
    gp_weight = 0.0  # Changed from 10.0 to 0.0 as per your config
    rgb_weight = 1.0
    thermal_weight = 1.0
    depth_weight = 1.0


@register_model("marlin_vit_base_ytf")
@Singleton
class MarlinVitBaseConfig(NoArgInit, SharedConfig, Downloadable):
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    decoder_embed_dim = 384
    decoder_depth = 4
    decoder_num_heads = 6
    full_model_url = "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_base_ytf.full.pt"
    encoder_model_url = "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_base_ytf.encoder.pt"


@register_model("marlin_vit_small_ytf")
@Singleton
class MarlinVitSmallConfig(NoArgInit, SharedConfig, Downloadable):
    encoder_embed_dim = 384
    encoder_depth = 12
    encoder_num_heads = 6
    decoder_embed_dim = 192
    decoder_depth = 4
    decoder_num_heads = 3
    full_model_url = \
        "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_small_ytf.full.pt"
    encoder_model_url = \
        "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_small_ytf.encoder.pt"


@register_model("marlin_vit_large_ytf")
@Singleton
class MarlinVitLargeConfig(NoArgInit, SharedConfig, Downloadable):
    encoder_embed_dim = 1024
    encoder_depth = 24
    encoder_num_heads = 16
    decoder_embed_dim = 512
    decoder_depth = 12
    decoder_num_heads = 8
    full_model_url = \
        "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_large_ytf.full.pt"
    encoder_model_url = \
        "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_large_ytf.encoder.pt"


@register_model("multimodal_marlin_base")
@Singleton
class MultiModalMarlinBaseConfig(NoArgInit, SharedMultiModalConfig):
    """Configuration for base size MultiModalMarlin"""
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    decoder_embed_dim = 384
    decoder_depth = 4  # Changed from 8 to 4 to match your configuration
    decoder_num_heads = 6


@register_model("multimodal_marlin_small")
@Singleton
class MultiModalMarlinSmallConfig(NoArgInit, SharedMultiModalConfig):
    """Configuration for small size MultiModalMarlin"""
    encoder_embed_dim = 384
    encoder_depth = 12
    encoder_num_heads = 6
    decoder_embed_dim = 192
    decoder_depth = 8
    decoder_num_heads = 3


@register_model("multimodal_marlin_large")
@Singleton
class MultiModalMarlinLargeConfig(NoArgInit, SharedMultiModalConfig):
    """Configuration for large size MultiModalMarlin"""
    encoder_embed_dim = 1024
    encoder_depth = 24
    encoder_num_heads = 16
    decoder_embed_dim = 512
    decoder_depth = 12
    decoder_num_heads = 8


@register_model("marlin_vit_base")
@Singleton
class MarlinVitBaseCustomConfig(NoArgInit, SharedConfig):
    """Configuration specifically matching your provided YAML setup"""
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    decoder_embed_dim = 384
    decoder_depth = 4
    decoder_num_heads = 6
    # This matches your specific configuration


def register_model_from_yaml(name: str, path: str, is_multimodal: bool = False) -> None:
    """
    Register a model configuration from a YAML file

    Args:
        name: Name to register the model configuration as
        path: Path to the YAML configuration file
        is_multimodal: Whether this is a MultiModalMarlin configuration
    """
    config = read_yaml(path)

    if is_multimodal:
        marlin_config = MultiModalMarlinConfig(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            n_frames=config["clip_frames"],
            encoder_embed_dim=config["encoder"]["embed_dim"],
            encoder_depth=config["encoder"]["depth"],
            encoder_num_heads=config["encoder"]["num_heads"],
            decoder_embed_dim=config["decoder"]["embed_dim"],
            decoder_depth=config["decoder"]["depth"],
            decoder_num_heads=config["decoder"]["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=config["qkv_bias"],
            qk_scale=config["qk_scale"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            norm_layer=config["norm_layer"],
            init_values=config["init_values"],
            tubelet_size=config["tubelet_size"],
            # Additional MultiModalMarlin params
            d_steps=config.get("d_steps", 3),
            g_steps=config.get("g_steps", 1),
            adv_weight=config.get("adv_weight", 0.1),
            gp_weight=config.get("gp_weight", 10.0),
            rgb_weight=config.get("rgb_weight", 1.0),
            thermal_weight=config.get("thermal_weight", 1.0),
            depth_weight=config.get("depth_weight", 1.0)
        )
    else:
        marlin_config = MarlinConfig(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            n_frames=config["clip_frames"],
            encoder_embed_dim=config["encoder"]["embed_dim"],
            encoder_depth=config["encoder"]["depth"],
            encoder_num_heads=config["encoder"]["num_heads"],
            decoder_embed_dim=config["decoder"]["embed_dim"],
            decoder_depth=config["decoder"]["depth"],
            decoder_num_heads=config["decoder"]["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=config["qkv_bias"],
            qk_scale=config["qk_scale"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            norm_layer=config["norm_layer"],
            init_values=config["init_values"],
            tubelet_size=config["tubelet_size"]
        )

    _configs[name] = marlin_config


def resolve_config(name: str) -> MarlinConfig:
    if name in _configs:
        return _configs[name]
    else:
        raise ValueError(f"Model {name} not found. Please register it first. The current registered models are: "
                         f"{list(_configs.keys())}")