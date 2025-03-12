import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.classifier import Classifier


class AttentionClassifier(Classifier):
    """
    A classifier that uses temporal self-attention to weight the encoded features
    for pain assessment from videos.

    This extends the base Classifier with an attention mechanism that learns
    to focus on the most pain-relevant frames in a video sequence.
    """

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
                 marlin_ckpt: str = None, task: str = "binary",
                 learning_rate: float = 1e-4, distributed: bool = False,
                 attention_dim: int = 64, num_heads: int = 1, dropout: float = 0.1):
        """
        Initialize the AttentionClassifier.

        Args:
            num_classes: Number of output classes
            backbone: Type of backbone to use (e.g., "marlin_vit_base_ytf")
            finetune: Whether to fine-tune the backbone
            marlin_ckpt: Path to a pre-trained MARLIN checkpoint
            task: Classification task type ("binary", "multiclass", "multilabel", "regression")
            learning_rate: Learning rate for optimization
            distributed: Whether to use distributed training
            attention_dim: Dimension of attention mechanism
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(num_classes, backbone, finetune, marlin_ckpt, task, learning_rate, distributed)

        # Get encoder embedding dimension from the model configuration
        config = self.hparams.get('config', None)
        if config:
            encoder_dim = config.encoder_embed_dim
        else:
            from model.config import resolve_config
            config = resolve_config(backbone)
            encoder_dim = config.encoder_embed_dim

        # Attention components
        if num_heads > 1:
            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=encoder_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            # Single-head attention
            self.query = nn.Linear(encoder_dim, attention_dim)
            self.key = nn.Linear(encoder_dim, attention_dim)
            self.value = nn.Linear(encoder_dim, encoder_dim)  # Value keeps original dimensionality

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Replace the original classification head
        # self.fc = nn.Sequential(
        #     nn.Linear(encoder_dim, encoder_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(encoder_dim // 2, num_classes)
        # )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(encoder_dim),  # Add BatchNorm before any transformations
            nn.Linear(encoder_dim, num_classes)  # Single layer classifier
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def apply_attention(self, x: Tensor) -> Tensor:
        """
        Apply temporal self-attention to the sequence of encoded clip features.

        Args:
            x: Input tensor of shape [batch_size, num_clips, feature_dim]
                where each element in the sequence represents features from a 16-frame clip

        Returns:
            Tensor: Attended features of shape [batch_size, feature_dim]
        """
        # Handle 4D input
        if len(x.shape) == 4:
            b, s1, s2, d = x.shape
            x = x.reshape(b, s1 * s2, d)  # Reshape to 3D
        if self.num_heads > 1:
            # Multi-head attention (using PyTorch's implementation)
            attended_output, _ = self.attention(x, x, x)

            # Pool across the sequence dimension
            attended_output = attended_output.mean(dim=1)
        else:
            # Single-head attention
            batch_size, seq_len, feature_dim = x.shape

            # Project inputs to query/key space
            queries = self.query(x)  # [batch_size, seq_len, attention_dim]
            keys = self.key(x)  # [batch_size, seq_len, attention_dim]
            values = self.value(x)  # [batch_size, seq_len, feature_dim]

            # Compute attention scores
            scores = torch.bmm(queries, keys.transpose(1, 2))  # [batch_size, seq_len, seq_len]
            scores = scores / (self.attention_dim ** 0.5)  # Scale scores

            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=2)  # [batch_size, seq_len, seq_len]

            # Add entropy regularization to promote more uniform attention
            # (prevents attention from focusing too much on specific frames)
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-7), dim=2).mean()
            self.log('attention_entropy', attention_entropy, on_step=False, on_epoch=True)

            # Apply attention weights to values
            attended_output = torch.bmm(attention_weights, values)  # [batch_size, seq_len, feature_dim]

            # Pool across the sequence dimension
            attended_output = attended_output.mean(dim=1)  # [batch_size, feature_dim]

        attended_output = self.dropout(attended_output)
        return attended_output

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor (video features)
               If using the Marlin encoder, this is expected to be a batch of videos
               where each video will be processed as a sequence of 16-frame clips

        Returns:
            Tensor: Class predictions
        """
        if self.model is not None:
            print("Using Marlin backbone for feature extraction now")
            # Get features from backbone, keeping sequence dimension
            features = self.model.extract_features(x, keep_seq=True)
        else:
            # If no backbone (using pre-extracted features)
            if len(x.shape) == 2:
                # Handle case where input is already pooled
                return self.fc(x)
            features = x

        # Apply attention to weight the most relevant frames
        attended_features = self.apply_attention(features)

        # Classification head
        output = self.fc(attended_features)
        return output

    def configure_optimizers(self):
        """
        Configure optimizers with layer-wise learning rate decay.
        Lower learning rate for backbone, higher for attention and classification layers.
        """
        if self.model is not None:
            # Two parameter groups with different learning rates
            backbone_params = list(self.model.parameters())
            attention_params = list(self.query.parameters()) + list(self.key.parameters()) + \
                               list(self.value.parameters()) + list(self.fc.parameters())

            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': self.learning_rate * 0.05},  # Lower LR for backbone
                {'params': attention_params, 'lr': self.learning_rate}  # Higher LR for new layers
            ], betas=(0.5, 0.9))
        else:
            # If using pre-extracted features
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8
                ),
                "monitor": "train_loss"
            }
        }