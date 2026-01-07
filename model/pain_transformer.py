import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm

# --- Helper Modules ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Temporal Transformer ---

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=384, num_latents=8, num_classes=5, embed_dim=384, cross_heads=1, self_heads=8, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        
        # Learnable query tokens
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)
        
        # Cross-attention layer
        self.cross_norm_q = nn.LayerNorm(embed_dim)
        self.cross_norm_k = nn.LayerNorm(input_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, cross_heads, dropout=dropout, batch_first=True)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Two self-attention layers
        self.self_layers = nn.ModuleList([
            Block(dim=embed_dim, num_heads=self_heads, mlp_ratio=4., drop=dropout, attn_drop=dropout)
            for _ in range(2)
        ])
        
        self.out_norm = nn.LayerNorm(embed_dim)

    def _sinusoidal_pos_encoding(self, T, dim, device):
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe = torch.zeros(T, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0) # (1, T, D)

    def forward(self, x):
        # x: (B, T, 384)
        B, T, D = x.shape
        
        # Fourier positional encoding
        pe = self._sinusoidal_pos_encoding(T, D, x.device)
        x = x + pe
        
        # Cross-attention
        # Query: latents (B, N, D)
        q = repeat(self.latents, 'n d -> b n d', b=B)
        
        # Pre-norm for cross-attn (following standard pre-norm transformer, or spec order)
        # Spec says: "Cross-attention layer ... Feed-forward network after each attention block"
        q_norm = self.cross_norm_q(q)
        k_norm = self.cross_norm_k(x)
        
        # MultiheadAttention(query, key, value)
        attn_out, _ = self.cross_attn(q_norm, k_norm, k_norm)
        q = q + attn_out
        
        # FFN
        q = q + self.cross_ff(q)
        
        # Self-attention layers
        for blk in self.self_layers:
            q = blk(q)
            
        # Output: Mean pooled video embedding
        q = self.out_norm(q)
        video_embed = q.mean(dim=1) # (B, 384)
        
        return video_embed


# --- Full Model ---

class PainEstimationModel(nn.Module):
    def __init__(self, num_classes=5, num_frames=16, spatial_drop_path=0.1):
        super().__init__()
        
        print("Initializing timm model: tnt_s_patch16_224 (ImageNet pretrained)...")
        # Using timm's TNT-Small (embed_dim=384)
        # num_classes=0 means no classification head (returns features/embedding)
        self.spatial_encoder = timm.create_model(
            'tnt_s_patch16_224', 
            pretrained=True, 
            num_classes=0, 
            drop_path_rate=spatial_drop_path
        )
        
        # Dimension is 384 for TNT-Small
        embed_dim = 384
        
        self.temporal_transformer = TemporalTransformer(
            input_dim=embed_dim,
            num_latents=8,
            embed_dim=embed_dim,
            cross_heads=1,
            self_heads=8,
            dropout=0.1
        )
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize custom weights (Temporal & Head)
        # Important: Do NOT re-init spatial_encoder as it has pretrained weights!
        self.temporal_transformer.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape
        
        # Fold time into batch for spatial encoding
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # Spatial Encode
        # timm model forward usually returns the embedding directly if num_classes=0
        frame_embeds = self.spatial_encoder(x) # (B*T, 384)
        
        # Unfold time
        frame_embeds = rearrange(frame_embeds, '(b t) d -> b t d', b=B, t=T)
        
        # Temporal Encode
        video_embed = self.temporal_transformer(frame_embeds) # (B, 384)
        
        # Classify
        logits = self.head(video_embed)
        
        return logits

    def load_pretrained(self, checkpoint_path: str):
        """Load weights from a PyTorch Lightning checkpoint."""
        print(f"Loading pretrained weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Adjust keys: remove 'model.' prefix if present (PL adds this)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
                
        # Handle head mismatch (e.g. BioVid might have diff classes than Syracuse)
        model_dict = self.state_dict()
        
        # Filter out head weights if shapes don't match
        if 'head.weight' in new_state_dict:
            if new_state_dict['head.weight'].shape != model_dict['head.weight'].shape:
                print(f"Skipping head weights due to shape mismatch: {new_state_dict['head.weight'].shape} vs {model_dict['head.weight'].shape}")
                del new_state_dict['head.weight']
                del new_state_dict['head.bias']
        
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def freeze_backbone(self):
        """Freeze Spatial Encoder and Temporal Transformer, keep Head trainable."""
        print("Freezing Spatial Encoder and Temporal Transformer...")
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
        for param in self.temporal_transformer.parameters():
            param.requires_grad = False
        # Ensure head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
