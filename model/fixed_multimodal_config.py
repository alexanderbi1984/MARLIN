from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class FixedMultiModalMarlinConfig:
    # Basic MARLIN parameters
    img_size: int = 224
    patch_size: int = 16
    n_frames: int = 16
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    decoder_embed_dim: int = 384
    decoder_depth: int = 4
    decoder_num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    norm_layer: str = "LayerNorm"
    init_values: float = 0.0
    tubelet_size: int = 2
    # MultiModal specific parameters
    d_steps: int = 1
    g_steps: int = 1
    adv_weight: float = 0.01
    gp_weight: float = 0.0
    rgb_weight: float = 1.0
    thermal_weight: float = 1.0
    depth_weight: float = 1.0

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with a default."""
        return getattr(self, key, default)

# Global instance - you can import this directly
CONFIG = FixedMultiModalMarlinConfig()

def get_config() -> FixedMultiModalMarlinConfig:
    """Get the configuration instance."""
    return CONFIG 