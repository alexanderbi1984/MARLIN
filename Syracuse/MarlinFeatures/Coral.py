import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CoralOrdinalMLPClassifier(torch.nn.Module):
    """
    CORAL-style Ordinal Regression with MLP backbone.
 
    Predicts ordinal labels using cumulative link modeling:
    For K classes, outputs K-1 sigmoid decisions: P(y > k), and counts how many are true.
 
    Args:
        input_dim (int): Dimensionality of input features.
        hidden_layers (List[int]): Hidden layer sizes for the MLP encoder.
        num_classes (int): Number of ordinal classes.
    """
    def __init__(self, input_dim, hidden_layers=[128, 64], num_classes=4):
        super().__init__()
        self.num_classes = num_classes
 
        # Define the MLP encoder with customizable hidden layers.
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = h
        self.encoder = torch.nn.Sequential(*layers)
 
        # Shared output logit.
        self.output_layer = torch.nn.Linear(prev_dim, 1)
 
        # Learnable thresholds for ordinal classes (K-1 thresholds).
        self.thresholds = torch.nn.Parameter(torch.arange(num_classes - 1, dtype=torch.float))
 
    def forward(self, x):
        """Forward pass producing cumulative probabilities P(y > k)."""
        x = self.encoder(x)
        logit = self.output_layer(x)  # shape: (batch_size, 1)
 
        # Ensure thresholds are ordered by applying softplus and cumulative sum.
        thresholds = torch.cumsum(torch.nn.functional.softplus(self.thresholds), dim=0)
 
        # Compute cumulative logits and probabilities.
        logits = logit - thresholds.view(1, -1)  # shape: (batch_size, K-1)
        probas = torch.sigmoid(logits)  # shape: (batch_size, K-1)
        return probas
 
    def predict_class(self, x):
        """
        Converts cumulative probabilities into ordinal class predictions.
 
        Args:
            x (torch.Tensor): Input features (batch_size, input_dim)
 
        Returns:
            torch.Tensor: Predicted class indices (batch_size,)
        """
        with torch.no_grad():
            cum_probs = self.forward(x)  # shape: (batch_size, K-1)
            # Count how many thresholds are passed (proba > 0.5) to determine the class.
            return torch.sum(cum_probs > 0.5, dim=1)