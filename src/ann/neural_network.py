"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import get_loss
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Build the network from CLI arguments.

        Expected cli_args attributes:
            - hidden_size: list of hidden layer sizes
            - num_layers: number of hidden layers
            - activation: activation function name
            - weight_init: weight initialization method
            - loss: loss function name
            - optimizer: optimizer name
            - learning_rate: learning rate
            - weight_decay: L2 regularization weight decay
            - dataset: dataset name (determines input/output dims)
        """
        self.args = cli_args

        # Determine layer sizes
        input_size = 784  # MNIST / Fashion-MNIST
        output_size = 10

        # Build hidden layer sizes list
        num_layers = getattr(cli_args, 'num_layers', 3)
        hidden_size = getattr(cli_args, 'hidden_size', [128])

        # If a single size is given, replicate for all hidden layers
        if len(hidden_size) == 1:
            hidden_sizes = hidden_size * num_layers
        else:
            hidden_sizes = hidden_size

        activation = getattr(cli_args, 'activation', 'relu')
        weight_init = getattr(cli_args, 'weight_init', 'xavier')

        # Build layers
        self.layers = []
        prev_size = input_size

        for h_size in hidden_sizes:
            layer = NeuralLayer(prev_size, h_size, activation=activation, weight_init=weight_init)
            self.layers.append(layer)
            prev_size = h_size

        # Output layer (no activation - returns logits)
        output_layer = NeuralLayer(prev_size, output_size, activation="softmax", weight_init=weight_init)
        self.layers.append(output_layer)

        # Loss function
        loss_name = getattr(cli_args, 'loss', 'cross_entropy')
        self.loss_fn = get_loss(loss_name)

        # Optimizer
        optimizer_name = getattr(cli_args, 'optimizer', 'adam')
        lr = getattr(cli_args, 'learning_rate', 0.001)
        wd = getattr(cli_args, 'weight_decay', 0.0)
        self.optimizer = get_optimizer(optimizer_name, self.layers, learning_rate=lr, weight_decay=wd)

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (pre-softmax linear combination from the last layer's z).
        X is shape (b, D_in) and output is shape (b, D_out).
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        # Return logits (the pre-activation z of the output layer)
        # The output layer applies softmax in forward, but we return z as logits
        self.logits = self.layers[-1].z
        return self.logits

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - grad_Ws[0] is gradient for the last (output) layer weights,
          grad_bs[0] is gradient for the last layer biases, and so on.
        """
        # Compute loss gradient w.r.t. logits
        # Loss functions internally apply softmax and compute combined gradient
        self.loss_fn.forward(y_pred, y_true)
        grad = self.loss_fn.backward()

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        for layer in reversed(self.layers):
            # For the output layer, skip the activation backward since loss already handles softmax
            if layer == self.layers[-1]:
                # Directly compute weight/bias gradients without activation backward
                layer.grad_W = layer.input.T @ grad
                layer.grad_b = np.sum(grad, axis=0, keepdims=True)
                grad = grad @ layer.W.T
            else:
                grad = layer.backward(grad)

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Create explicit object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        """Update weights using the optimizer."""
        self.optimizer.step()

    def compute_loss(self, logits, y_true_onehot):
        """Compute loss given logits and one-hot labels."""
        return self.loss_fn.forward(logits, y_true_onehot)

    def predict(self, X):
        """Return predicted class labels."""
        logits = self.forward(X)
        # Apply softmax to get probabilities
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def train_step(self, X_batch, y_batch_onehot):
        """Single training step on a batch."""
        logits = self.forward(X_batch)
        loss = self.compute_loss(logits, y_batch_onehot)
        self.backward(y_batch_onehot, logits)
        self.update_weights()
        return loss

    def evaluate(self, X, y):
        """
        Evaluate model on data.

        Args:
            X: Input data (N, 784)
            y: Integer labels (N,)

        Returns:
            accuracy: float
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_weights(self):
        """Get all model weights as a dictionary."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Set model weights from a dictionary."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
