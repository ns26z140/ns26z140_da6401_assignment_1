"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropyLoss:
    """Cross-Entropy loss for multi-class classification with softmax."""

    def forward(self, y_pred_logits, y_true):
        """
        Compute cross-entropy loss.
        y_pred_logits: raw logits (batch_size, num_classes)
        y_true: one-hot encoded labels (batch_size, num_classes)
        """
        # Apply softmax to logits for loss computation
        shifted = y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y_true = y_true

        # Clip to avoid log(0)
        eps = 1e-12
        clipped = np.clip(self.y_pred, eps, 1.0 - eps)
        loss = -np.sum(y_true * np.log(clipped)) / y_true.shape[0]
        return loss

    def backward(self):
        """
        Gradient of cross-entropy + softmax w.r.t. logits.
        dL/dz = (y_pred - y_true) / batch_size
        """
        return (self.y_pred - self.y_true)


class MeanSquaredErrorLoss:
    """Mean Squared Error loss."""

    def forward(self, y_pred_logits, y_true):
        """
        Compute MSE loss.
        y_pred_logits: raw logits (batch_size, num_classes)
        y_true: one-hot encoded labels (batch_size, num_classes)
        """
        # Apply softmax for MSE computation
        shifted = y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y_true = y_true
        self.logits = y_pred_logits

        loss = np.mean(np.sum((self.y_pred - y_true) ** 2, axis=1))
        return loss

    def backward(self):
        """
        Gradient of MSE loss w.r.t. logits (through softmax).
        dL/dz = (2/batch_size) * (y_pred - y_true) * y_pred * (1 - y_pred)
        More precisely, using the Jacobian of softmax.
        """
        batch_size = self.y_true.shape[0]
        num_classes = self.y_true.shape[1]

        diff = self.y_pred - self.y_true  # (batch_size, num_classes)

        # Gradient through softmax: for each sample, dL/dz_i = sum_j (dL/dy_j * dy_j/dz_i)
        # dy_j/dz_i = y_j*(delta_ij - y_i)
        grad = np.zeros_like(self.y_pred)
        for n in range(batch_size):
            y = self.y_pred[n].reshape(-1, 1)  # (C, 1)
            jacobian = np.diagflat(y) - y @ y.T  # (C, C)
            grad[n] = (2.0 * diff[n]) @ jacobian

        return grad


def get_loss(name):
    """Factory function to get loss function by name."""
    losses = {
        "cross_entropy": CrossEntropyLoss,
        "mean_squared_error": MeanSquaredErrorLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name]()
