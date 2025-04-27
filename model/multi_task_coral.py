import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim
from typing import Union, List

# --------------------------------------------------------------------
# Multi-Task CORAL Lightning Module
# --------------------------------------------------------------------

class MultiTaskCoralClassifier(pl.LightningModule):
    """
    Multi-task LightningModule using a shared encoder and two separate CORAL heads.
    Handles potentially missing labels for each task during training/evaluation.

    Args:
        input_dim (int): Dimension of the input features (e.g., 768).
        num_pain_classes (int): Number of ordinal classes for the pain level task (e.g., 5).
        num_stimulus_classes (int): Number of ordinal classes for the stimulus level task (e.g., 5).
        encoder_hidden_dims (list[int], optional): List of hidden dimensions for the shared MLP encoder. Defaults to None (Linear encoder).
        learning_rate (float, optional): Learning rate. Defaults to 1e-4.
        optimizer_name (str, optional): Optimizer name ('AdamW', 'Adam'). Defaults to 'AdamW'.
        pain_loss_weight (float, optional): Weight for the pain task loss. Defaults to 1.0.
        stim_loss_weight (float, optional): Weight for the stimulus task loss. Defaults to 1.0.
        # Add other hyperparameters like weight_decay if needed
    """
    def __init__(
        self,
        input_dim: int,
        num_pain_classes: int,
        num_stimulus_classes: int,
        encoder_hidden_dims: Union[List[int], None] = None,
        learning_rate: float = 1e-4,
        optimizer_name: str = 'AdamW',
        pain_loss_weight: float = 1.0,
        stim_loss_weight: float = 1.0,
        # weight_decay: float = 0.01 # Example
    ):
        super().__init__()

        if num_pain_classes <= 1 or num_stimulus_classes <= 1:
            raise ValueError("Number of classes for each task must be >= 2 for CORAL.")

        # Store hyperparameters
        self.save_hyperparameters()

        # --- Shared Encoder ---
        if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
            self.shared_encoder = nn.Linear(input_dim, input_dim) # Simple linear projection
            encoder_output_dim = input_dim
            # print(f"Using simple Linear layer as shared encoder (In: {input_dim}, Out: {encoder_output_dim})") # Optional logging
        else:
            layers = []
            current_dim = input_dim
            for h_dim in encoder_hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                current_dim = h_dim
            self.shared_encoder = nn.Sequential(*layers)
            encoder_output_dim = current_dim
            # print(f"Using MLP shared encoder (In: {input_dim}, Hidden: {encoder_hidden_dims}, Out: {encoder_output_dim})") # Optional logging


        # --- CORAL Heads ---
        self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
        self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)

        # --- Metrics ---
        # Using MAE as it's common for ordinal tasks, but others like Accuracy could be added
        metric_args = {'dist_sync_on_step': False} # Manage sync manually if needed or rely on Lightning
        self.train_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)

        self.train_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)

        # Add Accuracy if desired
        # self.train_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        # self.val_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        # self.test_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        # # ... and similar for stimulus task


    @staticmethod
    def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
        """Computes the CORAL loss (moved inside the class).

        Args:
            logits: Prediction logits of shape (batch_size, num_classes - 1).
            levels: Ground truth labels (levels) of shape (batch_size). Integer labels 0, 1, 2...
            importance_weights: Optional tensor of shape (batch_size) to weigh samples.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            The CORAL loss.
        """
        if logits.shape[0] == 0: # Handle empty batch
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Ensure levels are long integers
        levels = levels.long()

        # Create binary labels for each task t < num_classes - 1
        # levels_binary[b, t] = 1 if levels[b] > t else 0
        levels_binary = (levels.unsqueeze(1) > torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)).float()

        # Compute the loss for each task
        loss_tasks = F.logsigmoid(logits) * levels_binary + (F.logsigmoid(logits) - logits) * (1 - levels_binary)

        # Sum losses across tasks for each sample
        loss_per_sample = -torch.sum(loss_tasks, dim=1)

        if importance_weights is not None:
            loss_per_sample *= importance_weights

        if reduction == 'mean':
            return loss_per_sample.mean()
        elif reduction == 'sum':
            return loss_per_sample.sum()
        elif reduction == 'none':
            return loss_per_sample
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

    @staticmethod
    def prob_to_label(probs):
        """Converts predicted probabilities to labels (moved inside the class).

        Args:
            probs: Predicted probabilities of shape (batch_size, num_classes - 1).

        Returns:
            Predicted labels of shape (batch_size).
        """
        # probs[b, t] = P(y > t)
        # Label = sum_{t=0}^{num_classes-2} P(y > t) > 0.5
        # The result is an integer label 0, 1, ..., num_classes-1
        return torch.sum(probs > 0.5, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the encoder and heads. """
        encoded_features = self.shared_encoder(x)
        pain_logits = self.pain_head(encoded_features)
        stimulus_logits = self.stimulus_head(encoded_features)
        return pain_logits, stimulus_logits

    def _calculate_loss_and_metrics(self, pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage: str):
        """ Helper to calculate loss and update metrics for a given stage. """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Ensure loss is on correct device and requires grad
        pain_loss = torch.tensor(0.0, device=self.device)
        stim_loss = torch.tensor(0.0, device=self.device)

        # --- Pain Task ---
        valid_pain_mask = pain_labels != -1
        if valid_pain_mask.any():
            valid_pain_logits = pain_logits[valid_pain_mask]
            valid_pain_labels = pain_labels[valid_pain_mask]
            # Use static method via self
            pain_loss = self.coral_loss(valid_pain_logits, valid_pain_labels)

            # Update Metrics
            pain_probs = torch.sigmoid(valid_pain_logits)
            pain_preds = self.prob_to_label(pain_probs)
            mae_metric = getattr(self, f"{stage}_pain_mae")
            mae_metric.update(pain_preds, valid_pain_labels)
            
            self.log(f"{stage}_pain_loss", pain_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            # Restore original MAE log
            self.log(f"{stage}_pain_MAE", mae_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # --- Stimulus Task ---
        valid_stim_mask = stimulus_labels != -1
        if valid_stim_mask.any():
            valid_stim_logits = stimulus_logits[valid_stim_mask]
            valid_stim_labels = stimulus_labels[valid_stim_mask]
            # Use static method via self
            stim_loss = self.coral_loss(valid_stim_logits, valid_stim_labels)

            # Update Metrics
            stim_probs = torch.sigmoid(valid_stim_logits)
            stim_preds = self.prob_to_label(stim_probs)
            mae_metric = getattr(self, f"{stage}_stim_mae")
            mae_metric.update(stim_preds, valid_stim_labels)
            
            self.log(f"{stage}_stim_loss", stim_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            # Restore original MAE log
            self.log(f"{stage}_stim_MAE", mae_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # --- Combine Losses ---
        # Weighted sum using hyperparameters
        # Ensure requires_grad=True propagates if only one loss is active
        if valid_pain_mask.any() or valid_stim_mask.any():
             total_loss = self.hparams.pain_loss_weight * pain_loss + self.hparams.stim_loss_weight * stim_loss
        else:
             # No valid labels in batch, return zero loss but ensure grad is enabled if model parameters were used
             # This shouldn't happen if dataloaders guarantee at least one valid label per batch,
             # but handles the edge case. The forward pass still runs.
             total_loss = (pain_logits.sum() + stimulus_logits.sum()) * 0.0 # Trick to keep grad connection


        self.log(f"{stage}_loss", total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        loss = self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='train')
        # Log learning rate
        self.log("learning_rate", self.hparams.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("[DEBUG] Entering validation_step...")
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='val')
        # No loss returned for validation/test

    def test_step(self, batch, batch_idx):
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='test')
        # No loss returned for validation/test

    def sanity_check(self, batch_size: int = 4):
        """Performs a basic check on input/output shapes."""
        print(f"Running sanity check for {self.__class__.__name__}...")
        try:
            # Ensure model is on the correct device for the check
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.to(device)
            # Using self.device which Lightning manages
            device = self.device
            dummy_input = torch.randn(batch_size, self.hparams.input_dim, device=device)
            self.eval() # Set model to evaluation mode for the check
            with torch.no_grad(): # No need to track gradients
                pain_logits, stimulus_logits = self(dummy_input)
            self.train() # Set back to train mode

            # Check output shapes
            expected_pain_shape = (batch_size, self.hparams.num_pain_classes - 1)
            expected_stim_shape = (batch_size, self.hparams.num_stimulus_classes - 1)

            assert pain_logits.shape == expected_pain_shape, \
                f"Pain logits shape mismatch: Expected {expected_pain_shape}, Got {pain_logits.shape}"
            assert stimulus_logits.shape == expected_stim_shape, \
                f"Stimulus logits shape mismatch: Expected {expected_stim_shape}, Got {stimulus_logits.shape}"

            print(f"Sanity check passed for {self.__class__.__name__}.")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Pain logits shape: {pain_logits.shape}")
            print(f"  Stimulus logits shape: {stimulus_logits.shape}")

        except Exception as e:
            print(f"Sanity check failed for {self.__class__.__name__}: {e}")
            raise # Re-raise the exception after printing

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use."""
        # Simple optimizer for debugging
        # print("[DEBUG] Configuring simplified Adam optimizer...")
        # optimizer = optim.Adam(self.parameters(), lr=1e-4) 
        
        # --- Restore Original Code ---
        # Remove debug prints
        # print(f"[DEBUG configure_optimizers] Optimizer Name: {self.hparams.optimizer_name}, Type: {type(self.hparams.optimizer_name)}")
        # print(f"[DEBUG configure_optimizers] Learning Rate: {self.hparams.learning_rate}, Type: {type(self.hparams.learning_rate)}")
        
        # Explicitly cast learning rate to float
        try:
            lr = float(self.hparams.learning_rate)
        except ValueError:
            print(f"Error: Could not convert learning rate '{self.hparams.learning_rate}' to float!")
            raise # Re-raise the error
            
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr) # Use the float lr
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr) # Use the float lr
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        
        # Example LR scheduler (optional)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        # -- End Original Code ---
        return optimizer

# Remove the old if __name__ == '__main__': block if it exists
# (The edit tool should replace the entire file content based on the instruction) 