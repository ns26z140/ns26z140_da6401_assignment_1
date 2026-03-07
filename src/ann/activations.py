"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class Sigmoid:
    """Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))"""

    def forward(self, x):
        # Clip to avoid overflow in exp
        x = np.clip(x, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)


class Tanh:
    """Tanh activation: tanh(x)"""

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0).astype(np.float64)


class Softmax:
    """Softmax activation for output layer."""

    def forward(self, x):
        # Subtract max for numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # When used with cross-entropy, the combined gradient is (y_pred - y_true)
        # This is handled in the loss function, so just pass through
        return grad_output


def get_activation(name):
    """Factory function to get activation by name."""
    activations = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "softmax": Softmax,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]()
