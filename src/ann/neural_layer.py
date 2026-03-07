"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import get_activation


class NeuralLayer:
    """A single fully-connected layer with activation."""

    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier"):
        """
        Initialize a neural layer.

        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            activation: Activation function name
            weight_init: Weight initialization method ('random' or 'xavier')
        """
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        self.b = np.zeros((1, output_size))

        # Activation
        self.activation = get_activation(activation)
        self.activation_name = activation

        # Gradients (exposed for autograder verification)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, X):
        """
        Forward pass: z = X @ W + b, a = activation(z)

        Args:
            X: Input data of shape (batch_size, input_size)
        Returns:
            Activated output of shape (batch_size, output_size)
        """
        self.input = X  # Store for backward pass
        self.z = X @ self.W + self.b  # Pre-activation
        self.output = self.activation.forward(self.z)  # Post-activation
        return self.output

    def backward(self, grad_output):
        """
        Backward pass: compute gradients and return gradient w.r.t. input.

        Args:
            grad_output: Gradient from the next layer, shape (batch_size, output_size)
        Returns:
            Gradient w.r.t. input, shape (batch_size, input_size)
        """
        # Gradient through activation
        grad_z = self.activation.backward(grad_output)

        # Compute weight and bias gradients
        self.grad_W = self.input.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        # Gradient w.r.t. input for previous layer
        grad_input = grad_z @ self.W.T

        return grad_input
