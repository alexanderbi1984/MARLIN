import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - allow import where Lightning isn't available at import-time
    pl = object  # type: ignore


def sinusoidal_position_encoding(length: int, dim: int, device=None) -> Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, latents: Tensor, tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # latents: (B, L, D) -> (L, B, D)
        # tokens: (B, T, D) -> (T, B, D)
        q = self.ln_q(latents)
        kv = self.ln_kv(tokens)
        q = q.transpose(0, 1)
        k = kv.transpose(0, 1)
        v = kv.transpose(0, 1)
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        out = out.transpose(0, 1)
        return latents + self.drop(out)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x


class XformerAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 256,
        heads: int = 4,
        latents: int = 16,
        cross_layers: int = 1,
        self_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.latents = nn.Parameter(torch.randn(latents, embed_dim) * 0.02)
        self.cross_blocks = nn.ModuleList([CrossAttnBlock(embed_dim, heads, dropout=dropout) for _ in range(cross_layers)])
        self.self_blocks = nn.ModuleList([SelfAttnBlock(embed_dim, heads, dropout=dropout) for _ in range(self_layers)])
        self.pos_cache_len = 0
        self.pos_cache: Optional[Tensor] = None

    def _pos(self, t: int, dim: int, device) -> Tensor:
        if self.pos_cache is None or self.pos_cache_len < t or self.pos_cache.device != device or self.pos_cache.shape[-1] != dim:
            self.pos_cache = sinusoidal_position_encoding(t, dim, device=device)
            self.pos_cache_len = t
        return self.pos_cache[:t]

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, T, 768), mask: (B, T) True for valid
        B, T, _ = x.shape
        h = self.in_proj(x)
        # add positional encoding on valid tokens
        pos = self._pos(T, h.shape[-1], h.device)
        h = h + pos.unsqueeze(0)
        # key_padding_mask: True for padding positions
        kpm = None
        if mask is not None:
            kpm = (~mask).to(torch.bool)
        # cross-attend latents to tokens
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.cross_blocks:
            lat = blk(lat, h, key_padding_mask=kpm)
        # self-attend latents
        for blk in self.self_blocks:
            lat = blk(lat)
        # readout mean
        bag = lat.mean(dim=1)
        return bag


class SimpleAttnAggregator(nn.Module):
    def __init__(self, input_dim: int = 768, embed_dim: int = 256, dropout: float = 0.1, gated: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.gated = gated
        if gated:
            self.v = nn.Linear(embed_dim, 1, bias=False)
            self.u = nn.Linear(embed_dim, embed_dim)
            self.w = nn.Linear(embed_dim, embed_dim)
        else:
            self.v = nn.Linear(embed_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.proj(x)
        if self.gated:
            a = torch.tanh(self.u(h)) * torch.sigmoid(self.w(h))
            logits = self.v(a).squeeze(-1)
        else:
            logits = self.v(torch.tanh(h)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)
        bag = torch.sum(w.unsqueeze(-1) * h, dim=1)
        return bag


class MeanPoolAggregator(nn.Module):
    """Simple Mean Pooling Aggregator (Baseline / Ablation).
    Projects input to embed_dim and averages over the time dimension.
    Equivalent to instance-level training if bag_size=1.
    """

    def __init__(self, input_dim: int = 768, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, T, D)
        h = self.proj(x)
        h = self.act(h)
        h = self.drop(h)
        
        if mask is not None:
            # Zero out masked positions so they don't contribute to sum
            # mask is True for valid, False for pad
            h = h * mask.unsqueeze(-1).type_as(h)
            # Count valid elements per bag
            den = mask.sum(dim=1, keepdim=True).type_as(h).clamp_min(1.0)
            bag = h.sum(dim=1) / den
        else:
            bag = h.mean(dim=1)
        
        return bag



class LocalTemporalBlock(nn.Module):
    """Lightweight temporal pre-processing on (B, T, E).

    Modes:
      - conv: depthwise temporal Conv1d with residual
      - attn: small self-attention with local band mask (window ~= kernel_size)
    """

    def __init__(self, embed_dim: int, mode: str = "conv", kernel_size: int = 3, dropout: float = 0.0, attn_heads: int = 1):
        super().__init__()
        self.mode = str(mode).lower()
        self.kernel = int(max(1, kernel_size))
        if self.kernel % 2 == 0:
            self.kernel += 1  # enforce odd for symmetric local window
        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(embed_dim)
        if self.mode == "attn":
            heads = int(max(1, attn_heads))
            self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=float(dropout), batch_first=True)
            self.ff = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(float(dropout)),
            )
        else:
            # Depthwise temporal conv over (E channels, T length)
            # Input will be (B, E, T)
            self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=self.kernel, padding=self.kernel // 2, groups=embed_dim, bias=True)
            self.pw = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B, T, E); mask: (B, T) True for valid
        if x.numel() == 0:
            return x
        if self.mode == "attn":
            # Local band mask for attention (T,T): allow |i-j| <= w
            B, T, E = x.shape
            w = self.kernel // 2
            if T <= 0:
                return x
            # Build (T, T) mask once per call (on device/dtype of x)
            with torch.no_grad():
                idx = torch.arange(T, device=x.device)
                dist = (idx[None, :] - idx[:, None]).abs()
                attn_mask = (dist > w).to(x.dtype) * (-1e9)
            x0 = x
            x = self.norm(x)
            # MHA supports key_padding_mask (True for pad positions)
            key_padding_mask = ~mask if mask is not None else None
            out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = x0 + self.drop(out)
            # FFN
            x = x + self.ff(x)
            # Zero out padded positions to be safe
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            return x
        else:
            # Conv path
            x0 = x
            # Zero invalid timesteps to avoid leaking padding
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            x = self.norm(x)
            x = x.transpose(1, 2)  # (B, E, T)
            x = self.dw(x)
            x = self.pw(x)
            x = x.transpose(1, 2)  # (B, T, E)
            x = x0 + self.drop(x)
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            return x


class MMETransformerAggregator(nn.Module):
    """Transformer-based MIL aggregator compatible with the MME downstream checkpoint.

    This mirrors the `TransformerMILAggregator` used in `/downstream/mil_coral.py`
    so that we can load its `aggregator.*` state_dict directly.
    """

    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 256,
        heads: int = 4,
        latents: int = 16,
        self_layers: int = 2,
        cross_layers: int = 1,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        readout_mode: str = "mean",
        topk: int = 4,
        use_local_preproc: bool = False,
        local_type: str = "conv",
        local_kernel_size: int = 3,
        local_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_latents = int(latents)
        self.num_heads = int(heads)
        self.num_self_layers = int(max(0, self_layers))
        self.num_cross_layers = int(max(1, cross_layers))
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.readout_mode = str(readout_mode).lower()
        self.topk = max(1, int(topk))

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim) if self.input_dim != self.embed_dim else nn.Identity()
        self.token_ln = nn.LayerNorm(self.embed_dim)
        self.token_drop = nn.Dropout(self.dropout)

        # Optional local temporal block operating on (B, T, E)
        self.use_local_preproc = bool(use_local_preproc)
        if self.use_local_preproc:
            self.local_block = LocalTemporalBlock(
                embed_dim=self.embed_dim,
                mode=str(local_type),
                kernel_size=int(local_kernel_size),
                dropout=float(local_dropout),
                attn_heads=int(max(1, min(self.num_heads, 4))),
            )
        else:
            self.local_block = None

        # Learned latent queries (global, shared across batch)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.embed_dim) * 0.02)

        # Cross-attention blocks (pre-LN)
        self.cross_q_ln = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_cross_layers)])
        self.cross_kv_ln = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_cross_layers)])
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.attn_dropout, batch_first=True)
                for _ in range(self.num_cross_layers)
            ]
        )
        self.cross_drop = nn.Dropout(self.dropout)
        self.cross_ff_ln = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_cross_layers)])
        self.cross_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.embed_dim * 4, self.embed_dim),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_cross_layers)
            ]
        )

        # Self-attention on latents (pre-LN encoder blocks)
        self.self_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.embed_dim * 4,
                    dropout=self.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_self_layers)
            ]
        )
        self.self_drop = nn.Dropout(self.dropout)

        # For compatibility with temperature-annealing hooks in the original trainer
        self.attn_temp = 1.0

    @staticmethod
    def _sinusoidal_pos_emb(T: int, D: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Standard sine-cosine positional embeddings (T, D)."""
        pe = torch.zeros(T, D, device=device, dtype=dtype)
        position = torch.arange(0, T, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-(math.log(10000.0) / max(1, D))))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (T, D)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, T, Din), mask: (B, T) with True for valid tokens
        if mask is None:
            # assume all valid if mask not provided
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        B, T, _ = x.shape
        feats = x
        tokens = self.input_proj(feats)
        pe = self._sinusoidal_pos_emb(T, self.embed_dim, device=tokens.device, dtype=tokens.dtype)
        tokens = self.token_ln(tokens + pe.unsqueeze(0))
        tokens = self.token_drop(tokens)  # (B, T, E)
        if self.local_block is not None:
            tokens = self.local_block(tokens, mask)

        # Initialize latents (B, L, E)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention stack
        key_padding_mask = ~mask  # True where we want to mask (padding)
        attn_tokens: Optional[Tensor] = None
        if self.num_cross_layers > 0:
            attn_weights_last: Optional[Tensor] = None
            for i in range(self.num_cross_layers):
                q_norm = self.cross_q_ln[i](q)
                kv_norm = self.cross_kv_ln[i](tokens)
                out, attn_w = self.cross_attn[i](q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask)
                q = q + self.cross_drop(out)
                q = q + self.cross_ff[i](self.cross_ff_ln[i](q))
                attn_weights_last = attn_w

            if attn_weights_last is not None:
                attn_latent = attn_weights_last  # (B, L, T)
                attn_tokens = attn_latent.mean(dim=1)
                attn_tokens = attn_tokens.masked_fill(~mask, 0.0)
                denom = attn_tokens.sum(dim=1, keepdim=True).clamp_min(1e-6)
                attn_tokens = attn_tokens / denom

        # Self-attention on latents
        for blk in self.self_blocks:
            q = blk(q)
        q = self.self_drop(q)

        # Readout: mean over L latents by default (to match Syracuse usage)
        # We keep the "cls" / "topk" code paths mainly for compatibility but
        # Syracuse code path uses the mean readout.
        readout = self.readout_mode
        if readout == "cls" and q.shape[1] > 0:
            bag = q[:, 0, :]
        elif readout == "topk" and attn_tokens is not None:
            scores = attn_tokens.masked_fill(~mask, float("-inf"))
            k = min(self.topk, scores.shape[1])
            if k <= 0:
                bag = q.mean(dim=1)
            else:
                topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)
                valid_topk = torch.isfinite(topk_vals)
                gathered = torch.gather(
                    tokens,
                    1,
                    topk_idx.unsqueeze(-1).expand(-1, k, self.embed_dim),
                )
                masked_vals = topk_vals.masked_fill(~valid_topk, -1e9)
                weights = F.softmax(masked_vals, dim=1) * valid_topk.float()
                norm = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                weights = weights / norm
                bag = (gathered * weights.unsqueeze(-1)).sum(dim=1)
                empty = ~valid_topk.any(dim=1)
                if empty.any():
                    bag = bag.clone()
                    bag[empty] = q[empty].mean(dim=1)
        else:
            bag = q.mean(dim=1)

        return bag


class SharedCoralLayer(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, temperature: float = 1.0, init_bias: Optional[List[float]] = None):
        super().__init__()
        assert num_classes >= 2
        self.K = num_classes
        self.w = nn.Parameter(torch.zeros(in_dim))
        self.bias = nn.Parameter(torch.zeros(self.K - 1))
        if init_bias is not None and len(init_bias) == self.K - 1:
            with torch.no_grad():
                self.bias.copy_(torch.tensor(init_bias))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.tau = temperature

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, D) -> logits: (B, K-1)
        z = self.gamma * (h @ self.w)  # (B,)
        logits = (z.unsqueeze(-1) - self.bias) / max(self.tau, 1e-6)
        return logits


def coral_targets(classes: Tensor, num_classes: int) -> Tensor:
    # classes: (B,) int in [0, K-1]
    B = classes.shape[0]
    K = num_classes
    # t_k = 1 if y > k else 0, for k=0..K-2
    k = torch.arange(K - 1, device=classes.device).unsqueeze(0).expand(B, -1)
    y = classes.unsqueeze(1).expand_as(k)
    tgt = (y > k).float()
    return tgt


def quadratic_weighted_kappa(y_true: Tensor, y_pred: Tensor, num_classes: int) -> float:
    # y_true, y_pred: (N,) int
    with torch.no_grad():
        y_true = y_true.view(-1).to(torch.int64).cpu()
        y_pred = y_pred.view(-1).to(torch.int64).cpu()
        K = num_classes
        N = y_true.numel()
        if N == 0:
            return 0.0
        conf = torch.zeros((K, K), dtype=torch.float64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < K and 0 <= p < K:
                conf[t, p] += 1
        hist_true = conf.sum(dim=1)
        hist_pred = conf.sum(dim=0)
        # Expected matrix
        E = torch.outer(hist_true, hist_pred) / max(N, 1)
        # Quadratic weights
        W = torch.zeros((K, K), dtype=torch.float64)
        for i in range(K):
            for j in range(K):
                W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2 if K > 1 else 1)
        # Kappa
        num = (W * conf).sum()
        den = (W * E).sum().clamp(min=1e-12)
        kappa = 1.0 - (num / den)
        return float(kappa.item())


class MILCoralTransformer(pl.LightningModule):
    def __init__(
        self,
        # data/model dims
        input_dim: int = 768,
        embed_dim: int = 256,
        num_classes: int = 5,
        # aggregator
        attn_type: str = "xformer",  # 'xformer' | 'simple' | 'gated' | 'mme_xformer'
        xformer_heads: int = 4,
        xformer_latents: int = 16,
        xformer_cross_layers: int = 1,
        xformer_self_layers: int = 2,
        xformer_dropout: float = 0.1,
        # loss heads
        coral_alpha: float = 1.0,
        temperature: float = 1.0,
        ce_weight: float = 0.25,
        # optimization
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-2,
        # optional class weights (for CE); list of floats length K
        class_weights: Optional[List[float]] = None,
        # which head to use for metrics/prediction: "coral" | "ce"
        eval_head: str = "ce",
        aux_num_classes: Optional[int] = None,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.coral_alpha = coral_alpha
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_head = str(eval_head)
        self.register_buffer("ce_class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)
        self.aux_num_classes = int(aux_num_classes) if aux_num_classes is not None else None
        self.aux_loss_weight = float(aux_loss_weight) if aux_loss_weight is not None else 0.0

        if attn_type == "xformer":
            self.aggregator = XformerAggregator(
                input_dim=input_dim,
                embed_dim=embed_dim,
                heads=xformer_heads,
                latents=xformer_latents,
                cross_layers=xformer_cross_layers,
                self_layers=xformer_self_layers,
                dropout=xformer_dropout,
            )
        elif attn_type == "mme_xformer":
            # MME-style Transformer MIL aggregator (for loading external pretrained checkpoints)
            self.aggregator = MMETransformerAggregator(
                input_dim=input_dim,
                embed_dim=embed_dim,
                heads=xformer_heads,
                latents=xformer_latents,
                self_layers=xformer_self_layers,
                cross_layers=xformer_cross_layers,
                dropout=xformer_dropout,
                attn_dropout=xformer_dropout,
            )
        elif attn_type == "simple":
            self.aggregator = SimpleAttnAggregator(input_dim=input_dim, embed_dim=embed_dim, dropout=xformer_dropout, gated=False)
        elif attn_type == "gated":
            self.aggregator = SimpleAttnAggregator(input_dim=input_dim, embed_dim=embed_dim, dropout=xformer_dropout, gated=True)
        elif attn_type == "mean":
            self.aggregator = MeanPoolAggregator(input_dim=input_dim, embed_dim=embed_dim, dropout=xformer_dropout)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.coral_head = SharedCoralLayer(in_dim=embed_dim, num_classes=num_classes, temperature=temperature)
        # CE head: simplified to a single linear layer for better regularization on small Syracuse dataset
        self.ce_head = nn.Linear(embed_dim, num_classes) if ce_weight and ce_weight > 0 else None
        self.aux_ce_head = nn.Linear(embed_dim, self.aux_num_classes) if self.aux_num_classes is not None and self.aux_loss_weight > 0 else None

        # Buffers for metrics aggregation
        self._train_preds: List[int] = []
        self._train_targets: List[int] = []
        self._train_abserr: List[float] = []
        self._val_preds: List[int] = []
        self._val_targets: List[int] = []
        self._val_abserr: List[float] = []

    @classmethod
    def from_mme_checkpoint(
        cls,
        ckpt_path: str,
        num_classes: Optional[int] = None,
        coral_alpha: float = 0.0,
        ce_weight: float = 1.0,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-2,
        class_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> "MILCoralTransformer":
        """Build a model with an MME-pretrained Transformer MIL aggregator.

        The aggregator weights are loaded from the given checkpoint's `aggregator.*`
        keys. The CE head is (re)initialized in this project and trained on Syracuse.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hp = ckpt.get("hyper_parameters", {})
        input_dim = int(hp.get("input_dim", 768))
        embed_dim = int(hp.get("embed_dim", 256))
        heads = int(hp.get("xformer_heads", 4))
        latents = int(hp.get("xformer_latents", 16))
        cross_layers = int(hp.get("xformer_cross_layers", 1))
        self_layers = int(hp.get("xformer_self_layers", 2))
        xformer_dropout = float(hp.get("xformer_dropout", 0.1))

        if num_classes is None:
            num_classes = int(hp.get("num_classes", 5))

        model = cls(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            attn_type="mme_xformer",
            xformer_heads=heads,
            xformer_latents=latents,
            xformer_cross_layers=cross_layers,
            xformer_self_layers=self_layers,
            xformer_dropout=xformer_dropout,
            coral_alpha=coral_alpha,
            ce_weight=ce_weight,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=class_weights,
            eval_head="ce",
            **kwargs,
        )

        state_dict = ckpt.get("state_dict", ckpt)
        agg_prefix = "aggregator."
        agg_sd = {k[len(agg_prefix) :]: v for k, v in state_dict.items() if k.startswith(agg_prefix)}
        missing, unexpected = model.aggregator.load_state_dict(agg_sd, strict=False)
        if missing:
            print(f"[from_mme_checkpoint] Warning: missing aggregator keys: {missing}")
        if unexpected:
            print(f"[from_mme_checkpoint] Warning: unexpected aggregator keys: {unexpected}")
        return model

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        bag = self.aggregator(x, mask)
        coral_logits = self.coral_head(bag)
        out: Dict[str, Tensor] = {"coral_logits": coral_logits}
        if self.ce_head is not None:
            out["ce_logits"] = self.ce_head(bag)
        out["bag"] = bag
        return out

    def coral_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (B, K-1); targets: ints (B,)
        # filter invalid labels
        K = self.num_classes
        valid = (targets >= 0) & (targets < K)
        if valid.any():
            logits_v = logits[valid]
            targets_v = targets[valid]
            tgt = coral_targets(targets_v, K)
            loss = F.binary_cross_entropy_with_logits(logits_v, tgt, reduction="mean")
            return loss
        # no valid labels in batch
        return logits.sum() * 0.0

    def ce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # filter invalid labels
        K = logits.shape[-1]
        valid = (targets >= 0) & (targets < K)
        if not valid.any():
            return logits.sum() * 0.0
        logits_v = logits[valid]
        targets_v = targets[valid]
        if self.ce_class_weights is not None:
            return F.cross_entropy(logits_v, targets_v, weight=self.ce_class_weights)
        return F.cross_entropy(logits_v, targets_v)

    def aux_ce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.aux_num_classes is None:
            return logits.sum() * 0.0
        K = self.aux_num_classes
        valid = (targets >= 0) & (targets < K)
        if not valid.any():
            return logits.sum() * 0.0
        logits_v = logits[valid]
        targets_v = targets[valid]
        return F.cross_entropy(logits_v, targets_v)

    @staticmethod
    def coral_predict_class(logits: Tensor) -> Tensor:
        # Predict class by counting thresholds with sigmoid(logit) > 0.5
        probs = torch.sigmoid(logits)
        pred = torch.sum(probs > 0.5, dim=1)
        return pred

    @staticmethod
    def coral_class_probs(logits: Tensor, num_classes: int) -> Tensor:
        """Convert CORAL logits (B, K-1) to class probabilities (B, K).
        Using g_k = P(y > k) = sigmoid(logit_k):
          P(y=0) = 1 - g_0
          P(y=c) = g_{c-1} - g_{c} for c=1..K-2
          P(y=K-1) = g_{K-2}
        """
        B = logits.shape[0]
        K = int(num_classes)
        if K <= 1:
            return torch.ones((B, 1), device=logits.device)
        g = torch.sigmoid(logits)  # (B, K-1)
        probs_list = []
        # class 0
        p0 = 1.0 - g[:, 0]
        probs_list.append(p0)
        # middle classes
        for c in range(1, K - 1):
            pc = g[:, c - 1] - g[:, c]
            probs_list.append(pc)
        # last class
        plast = g[:, K - 2]
        probs_list.append(plast)
        probs = torch.stack(probs_list, dim=1)
        # numerical safety
        probs = torch.clamp(probs, min=0.0, max=1.0)
        # normalize to sum=1
        s = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs = probs / s
        return probs

    @staticmethod
    def ce_predict_class(logits: Tensor) -> Tensor:
        """Argmax over CE logits to obtain class predictions."""
        return torch.argmax(logits, dim=-1)

    def _eval_logits_and_pred(self, out: Dict[str, Tensor]) -> Tensor:
        """Select which head to use for metrics based on `self.eval_head`."""
        head = getattr(self, "eval_head", "coral")
        if head == "ce" and self.ce_head is not None and ("ce_logits" in out):
            logits = out["ce_logits"]
            pred = self.ce_predict_class(logits)
        else:
            logits = out["coral_logits"]
            pred = self.coral_predict_class(logits)
        return pred.detach()

    def _primary_training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        x, mask, y = batch["x"], batch["mask"], batch["y"]
        out = self(x, mask)
        loss_coral = self.coral_loss(out["coral_logits"], y)
        loss = self.coral_alpha * loss_coral
        self.log("train_coral_loss", loss_coral, prog_bar=False, on_step=True, on_epoch=True)
        if self.ce_head is not None:
            loss_ce = self.ce_loss(out["ce_logits"], y)
            self.log("train_ce_loss", loss_ce, prog_bar=False, on_step=True, on_epoch=True)
            loss = loss + self.ce_weight * loss_ce
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Collect training metrics (per-epoch)
        pred = self._eval_logits_and_pred(out)
        K = self.num_classes
        valid = (y >= 0) & (y < K)
        yv = y[valid]
        pv = pred[valid]
        self._train_preds.extend([int(p.item()) for p in pv])
        self._train_targets.extend([int(t.item()) for t in yv])
        self._train_abserr.extend([abs(float(p) - float(t)) for p, t in zip(pv, yv)])
        return loss

    def _aux_step(self, batch: Dict[str, Tensor], stage: str = "train") -> Tensor:
        if self.aux_ce_head is None or self.aux_num_classes is None:
            return torch.tensor(0.0, device=self.device)
        if isinstance(batch, dict):
            x, mask, y = batch["x"], batch.get("mask"), batch["y"]
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                x = batch[0]
                y = batch[1]
            else:
                raise ValueError("Aux batch tuple must contain at least (x, y)")
            mask = batch[2] if len(batch) > 2 else None
        else:
            raise ValueError("Unsupported batch type for auxiliary step")
        bag = self.aggregator(x, mask)
        logits = self.aux_ce_head(bag)
        loss = self.aux_ce_loss(logits, y)
        self.log(f"{stage}_aux_loss", loss, prog_bar=False, on_step=(stage == "train"), on_epoch=True)
        with torch.no_grad():
            K = self.aux_num_classes
            valid = (y >= 0) & (y < K)
            if valid.any():
                preds = torch.argmax(logits[valid], dim=-1)
                acc = (preds == y[valid]).float().mean()
                self.log(f"{stage}_aux_acc", acc, prog_bar=False, on_step=(stage == "train"), on_epoch=True)
            else:
                self.log(f"{stage}_aux_acc", torch.tensor(0.0, device=self.device), prog_bar=False, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        if isinstance(batch, dict) and "x" not in batch:
            total_loss: Optional[Tensor] = None
            if "syracuse" in batch:
                primary_loss = self._primary_training_step(batch["syracuse"])
                total_loss = primary_loss if total_loss is None else total_loss + primary_loss
            if self.aux_ce_head is not None:
                aux_keys = [k for k in batch.keys() if k != "syracuse"]
                for key in aux_keys:
                    aux_loss = self._aux_step(batch[key], stage="train")
                    weighted = self.aux_loss_weight * aux_loss
                    total_loss = weighted if total_loss is None else total_loss + weighted
            if total_loss is None:
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
            return total_loss
        return self._primary_training_step(batch)

    def on_train_epoch_end(self) -> None:
        if len(self._train_preds) == 0:
            return
        preds = torch.tensor(self._train_preds)
        targs = torch.tensor(self._train_targets)
        qwk = quadratic_weighted_kappa(targs, preds, self.num_classes)
        acc = (preds == targs).float().mean().item()
        mae = (torch.tensor(self._train_abserr).float().mean().item()) if len(self._train_abserr) else 0.0
        self.log("train_qwk", qwk, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_mae", mae)
        # reset
        self._train_preds.clear()
        self._train_targets.clear()
        self._train_abserr.clear()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        if isinstance(batch, dict) and "x" not in batch:
            batch = batch.get("syracuse") or next(iter(batch.values()))
        x, mask, y = batch["x"], batch["mask"], batch["y"]
        out = self(x, mask)
        loss = 0.0
        if self.coral_alpha != 0.0:
            loss_coral = self.coral_loss(out["coral_logits"], y)
            loss = loss + self.coral_alpha * loss_coral
        if self.ce_head is not None and self.ce_weight != 0.0:
            loss_ce = self.ce_loss(out.get("ce_logits"), y)
            loss = loss + self.ce_weight * loss_ce
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # metrics
        pred = self._eval_logits_and_pred(out)
        # only collect valid labels
        K = self.num_classes
        valid = (y >= 0) & (y < K)
        yv = y[valid]
        pv = pred[valid]
        self._val_preds.extend([int(p.item()) for p in pv])
        self._val_targets.extend([int(t.item()) for t in yv])
        self._val_abserr.extend([abs(float(p) - float(t)) for p, t in zip(pv, yv)])

    def on_validation_epoch_end(self) -> None:
        if len(self._val_preds) == 0:
            return
        preds = torch.tensor(self._val_preds)
        targs = torch.tensor(self._val_targets)
        qwk = quadratic_weighted_kappa(targs, preds, self.num_classes)
        acc = (preds == targs).float().mean().item()
        mae = (torch.tensor(self._val_abserr).float().mean().item()) if len(self._val_abserr) else 0.0
        self.log("val_qwk", qwk, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_mae", mae)
        # reset
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_abserr.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=20, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_qwk"}}

    # -------------------------
    # Test loop (mirrors validation)
    # -------------------------
    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        if isinstance(batch, dict) and "x" not in batch:
            batch = batch.get("syracuse") or next(iter(batch.values()))
        x, mask, y = batch["x"], batch["mask"], batch["y"]
        out = self(x, mask)
        loss = 0.0
        if self.coral_alpha != 0.0:
            loss_coral = self.coral_loss(out["coral_logits"], y)
            loss = loss + self.coral_alpha * loss_coral
        if self.ce_head is not None and self.ce_weight != 0.0:
            loss_ce = self.ce_loss(out.get("ce_logits"), y)
            loss = loss + self.ce_weight * loss_ce
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        # metrics
        pred = self._eval_logits_and_pred(out)
        # reuse buffers for simplicity, only valid labels
        K = self.num_classes
        valid = (y >= 0) & (y < K)
        yv = y[valid]
        pv = pred[valid]
        self._val_preds.extend([int(p.item()) for p in pv])
        self._val_targets.extend([int(t.item()) for t in yv])
        self._val_abserr.extend([abs(float(p) - float(t)) for p, t in zip(pv, yv)])

    def on_test_epoch_end(self) -> None:
        if len(self._val_preds) == 0:
            return
        preds = torch.tensor(self._val_preds, dtype=torch.int64)
        targs = torch.tensor(self._val_targets, dtype=torch.int64)
        K = self.num_classes

        # Basic metrics
        qwk = quadratic_weighted_kappa(targs, preds, K)
        acc = (preds == targs).float().mean().item()
        mae = (torch.tensor(self._val_abserr).float().mean().item()) if len(self._val_abserr) else 0.0
        self.log("test_qwk", qwk, prog_bar=True)
        self.log("test_acc", acc)
        self.log("test_mae", mae)

        # Confusion matrix (rows=true, cols=pred)
        conf = torch.zeros((K, K), dtype=torch.int64)
        for t, p in zip(targs, preds):
            if 0 <= t < K and 0 <= p < K:
                conf[int(t), int(p)] += 1

        # Per-class precision/recall/F1
        tp = conf.diag().to(torch.float64)
        pred_pos = conf.sum(dim=0).to(torch.float64)  # column sums
        true_pos = conf.sum(dim=1).to(torch.float64)  # row sums
        precision = torch.where(pred_pos > 0, tp / pred_pos, torch.zeros_like(tp))
        recall = torch.where(true_pos > 0, tp / true_pos, torch.zeros_like(tp))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
        macro_f1 = f1.mean().item() if K > 0 else 0.0
        weighted_f1 = (f1 * true_pos / max(true_pos.sum().item(), 1.0)).sum().item()
        # Micro-F1 for single-label multiclass equals accuracy, but compute explicitly
        fp = pred_pos - tp
        fn = true_pos - tp
        micro_f1 = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())).item() if (2 * tp.sum() + fp.sum() + fn.sum()) > 0 else 0.0

        self.log("test_f1_macro", macro_f1)
        self.log("test_f1_weighted", weighted_f1)
        self.log("test_f1_micro", micro_f1)

        # Pretty print confusion matrix and class metrics
        print("\n===== Test Confusion Matrix (rows=true, cols=pred) =====")
        for i in range(K):
            row = " ".join(f"{int(v):5d}" for v in conf[i].tolist())
            print(f"class {i:2d}: {row}")
        print("=======================================================\n")

        print("Per-class metrics:")
        header = f"{'class':>5} {'support':>8} {'pred_pos':>8} {'prec':>7} {'recall':>7} {'f1':>7}"
        print(header)
        for i in range(K):
            sup = int(true_pos[i].item())
            pp = int(pred_pos[i].item())
            print(f"{i:5d} {sup:8d} {pp:8d} {precision[i].item():7.3f} {recall[i].item():7.3f} {f1[i].item():7.3f}")
        print("\nSummary: acc={:.3f} | qwk={:.3f} | mae={:.3f} | f1_macro={:.3f} | f1_weighted={:.3f} | f1_micro={:.3f}".format(
            acc, qwk, mae, macro_f1, weighted_f1, micro_f1
        ))
        # reset
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_abserr.clear()

    # -------------------------
    # Predict loop to export per-sample outputs
    # -------------------------
    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0):
        x, mask, y = batch["x"], batch["mask"], batch["y"]
        out = self(x, mask)
        head = getattr(self, "eval_head", "coral")
        if head == "ce" and self.ce_head is not None and ("ce_logits" in out):
            logits = out["ce_logits"]
            pred = self.ce_predict_class(logits).detach().cpu()
            probs = torch.softmax(logits, dim=-1).detach().cpu()
        else:
            logits = out["coral_logits"]
            pred = self.coral_predict_class(logits).detach().cpu()
            probs = self.coral_class_probs(logits, self.num_classes).detach().cpu()
        return {
            "y_true": y.detach().cpu(),
            "y_pred": pred,
            "y_probs": probs,
            "video_ids": batch.get("video_ids", None),
            "combos": batch.get("combos", None),
        }
        self._aux_train_preds.extend([int(p.item()) for p in preds[valid]])
        self._aux_train_targets.extend([int(t.item()) for t in targets[valid]])
