from typing import Optional, Union, Sequence, Dict, Literal, Any

from pytorch_lightning import LightningModule
from torch import Tensor, softmax
from torch.nn import CrossEntropyLoss, Linear, Identity, BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
# from marlin_pytorch import Marlin
# from marlin_pytorch.config import resolve_config
from model.marlin import Marlin
from model.config import resolve_config


class Classifier(LightningModule):

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
        marlin_ckpt: Optional[str] = None,
        task: Literal["binary", "multiclass", "multilabel", "regression"] = "binary",
        learning_rate: float = 1e-4, distributed: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        if finetune:
            if marlin_ckpt is None:
                self.model = Marlin.from_online(backbone).encoder
            else:
                self.model = Marlin.from_file(backbone, marlin_ckpt).encoder
        else:
            self.model = None

        config = resolve_config(backbone)

        self.fc = Linear(config.encoder_embed_dim, num_classes)
        self.learning_rate = learning_rate
        self.distributed = distributed
        self.task = task
        if task in "binary":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task=task, num_classes=1)
            self.auc_fn = AUROC(task=task, num_classes=1)
        elif task == "multiclass":
            self.loss_fn = CrossEntropyLoss()
            self.acc_fn = Accuracy(task=task, num_classes=num_classes)
            self.auc_fn = AUROC(task=task, num_classes=num_classes)
        elif task == "multilabel":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task="binary", num_classes=1)
            self.auc_fn = AUROC(task="binary", num_classes=1)
        elif task == "regression":
            self.loss_fn = MSELoss()  # You can also use L1Loss for MAE
            self.mse_fn = MeanSquaredError()
            self.mae_fn = MeanAbsoluteError()
            self.r2_fn = R2Score()

    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x):
        if self.model is not None:
            feat = self.model.extract_features(x, True)
        else:
            feat = x
        return self.fc(feat)

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x, y = batch
        # print(f"the shape of x is {x.shape}")
        # print(f"the shape of y is {y.shape}")
        # print(f"y in forward step is {y}")
        y_hat = self(x)
        # print(f"y_hat in forward step is {y_hat}")
        # print(f"y in forward step is {y}")
        # # Ensure y is of type Long for CrossEntropyLoss
        # y = y.long()  # Convert y to Long type
        # if self.task == "multilabel":
        #     y_hat = y_hat.flatten()
        #     y = y.flatten()
        # # loss = self.loss_fn(y_hat, y.float())
        # loss = self.loss_fn(y_hat, y)
        # Ensure y is of type Long for CrossEntropyLoss
        # y = y.long()  # Convert y to Long type for multi-class classification
        # if self.task == "binary":
        #     prob= y_hat.sigmoid()  # Apply sigmoid for binary
        # if self.task == "multilabel" :
        #     y_hat = y_hat.flatten()
        #     y = y.flatten()
        #     prob = y_hat.sigmoid() # Apply sigmoid for multilabel
        # if self.task == "multiclass":
        #     prob = softmax(y_hat, dim=1)  # Apply softmax for multiclass
        # loss = self.loss_fn(y_hat, y)
        # if self.task == "binary":
        #     # Use BCEWithLogitsLoss or BCE
        #     # In this case, we can use BCEWithLogitsLoss and pass y_hat directly
        #     loss = self.loss_fn(y_hat, y.float())  # If using BCEWithLogitsLoss
        #     prob = y_hat.sigmoid()  # Apply sigmoid to get probabilities

        if self.task == "binary":
            # Ensure y is of type Long for CrossEntropyLoss or float for BCE
            y = y.float()  # Ensure y is float for BCEWithLogitsLoss
            loss = self.loss_fn(y_hat.squeeze(), y)  # Squeeze to remove the second dimension
            prob = y_hat.sigmoid().squeeze()  # Apply sigmoid to get probabilities
            acc = self.acc_fn(prob, y)
            auc = self.auc_fn(prob, y)
            return {"loss": loss, "acc": acc, "auc": auc}

        elif self.task == "multilabel":
            y_hat = y_hat.flatten()
            y = y.flatten()
            loss = self.loss_fn(y_hat, y.float())  # Ensure loss function is appropriate
            prob = y_hat.sigmoid()  # Apply sigmoid for multilabel
            acc = self.acc_fn(prob, y)
            auc = self.auc_fn(prob, y)
            return {"loss": loss, "acc": acc, "auc": auc}

        elif self.task == "multiclass":
            # print(f"the shape of y_hat is {y_hat.shape}")
            # print(f"the shape of y is {y.shape}")
            # print(f"y_hat is {y_hat}")
            # print(f"y is {y}")
            loss = self.loss_fn(y_hat, y)  # Use appropriate loss for multiclass (CrossEntropyLoss)
            prob = softmax(y_hat, dim=1)  # Apply softmax for multiclass
            acc = self.acc_fn(prob, y)
            auc = self.auc_fn(prob, y)
            return {"loss": loss, "acc": acc, "auc": auc}
        elif self.task == "regression":
            y = y.float()
            y_hat = y_hat.squeeze()
            loss = self.loss_fn(y_hat, y)
            mse = self.mse_fn(y_hat, y)
            mae = self.mae_fn(y_hat, y)
            r2 = self.r2_fn(y_hat, y)
            return {"loss": loss, "mse": mse, "mae": mae, "r2": r2}



    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log("learning_rate", self.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }
