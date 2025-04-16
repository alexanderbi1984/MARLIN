from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.dataset import proba_to_label
from coral_pytorch.layers import CoralLayer

from typing import Optional, Union, Sequence, Dict, Literal, Any
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


from pytorch_lightning import LightningModule
import torchmetrics
import torch
import torch.nn as nn


# LightningModule that receives a PyTorch model as input
class LightningMLP(LightningModule):
    def __init__(self,  num_classes: int, 
        learning_rate: float = 1e-4,
        input_dim: int = 768,
        hidden_dim: int = 256,
        distributed: bool = False):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.distributed = distributed
        # Save settings and hyperparameters
        # Important: Add input_dim and hidden_dim if you want them logged
        self.save_hyperparameters("num_classes", "learning_rate", "input_dim", "hidden_dim")

        # --- Define the MLP Layers ---
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Example dropout
            CoralLayer(size_in=hidden_dim, num_classes=num_classes) # Output for CORAL is num_classes - 1
        )
        # ---------------------------

        # Set up metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        # Add Accuracy metric
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) # Use num_classes for accuracy
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    # --- Define the forward method ---
    def forward(self, x):
        return self.model(x)
    # ---------------------------------

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch

        # Convert class labels for CORAL ------------------------
        levels = levels_from_labelbatch(
            true_labels, num_classes=self.num_classes)
        # -------------------------------------------------------

        logits = self(features)

        # CORAL Loss --------------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, true_labels)
        loss = coral_loss(logits, levels.type_as(logits))
        # -------------------------------------------------------

        # CORAL Prediction to label -----------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        probas = torch.sigmoid(logits)
        predicted_labels = proba_to_label(probas)
        # -------------------------------------------------------

        # Calculate metrics (ensure device consistency if needed, though Lightning handles it)
        mae = self.train_mae(predicted_labels, true_labels) # Use the metric instance
        acc = self.train_acc(predicted_labels, true_labels) # Use the metric instance

        # Note: You might want separate metric instances for train/val/test
        # Updated above to reflect this common practice

        # Return loss and metrics
        return {"loss": loss, "mae": mae, "acc": acc} # Return metrics calculated by torchmetrics

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss_dict = self._shared_step(batch)
        # Log metrics using the correct instances
        self.log("train_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", self.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        # --- Recalculate metrics using validation instances ---
        features, true_labels = batch
        levels = levels_from_labelbatch(true_labels, num_classes=self.num_classes)
        logits = self(features)
        loss = coral_loss(logits, levels.type_as(logits))
        probas = torch.sigmoid(logits)
        predicted_labels = proba_to_label(probas)

        self.valid_mae.update(predicted_labels, true_labels)
        self.valid_acc.update(predicted_labels, true_labels)
        # -----------------------------------------------------

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", self.valid_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # --- Recalculate metrics using test instances ---
        features, true_labels = batch
        levels = levels_from_labelbatch(true_labels, num_classes=self.num_classes)
        logits = self(features)
        loss = coral_loss(logits, levels.type_as(logits))
        probas = torch.sigmoid(logits)
        predicted_labels = proba_to_label(probas)

        self.test_mae.update(predicted_labels, true_labels)
        self.test_acc.update(predicted_labels, true_labels)
        # ------------------------------------------------

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # self.parameters() should now find the parameters from self.model
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "val_mae" # Monitor val_mae now
            }
        }
