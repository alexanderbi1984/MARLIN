from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.dataset import proba_to_label
from typing import Optional, Union, Sequence, Dict, Literal, Any
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


from pytorch_lightning import LightningModule
import torchmetrics


# LightningModule that receives a PyTorch model as input
class LightningMLP(LightningModule):
    def __init__(self,  num_classes: int, 
        learning_rate: float = 1e-4, distributed: bool = False):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        # The inherited PyTorch module
        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters()

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()


    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
   
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
        mae = torchmetrics.MeanAbsoluteError(predicted_labels, true_labels)
        mse = torchmetrics.MeanSquaredError(predicted_labels, true_labels)
        r2 = torchmetrics.R2Score(predicted_labels, true_labels)
        return {"loss": loss, "mae": mae, "mse": mse, "r2": r2}

    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss_dict = self._shared_step(batch)
        self.log("learning_rate", self.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx)-> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict({f"test_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }
