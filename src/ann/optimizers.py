"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

import numpy as np


class SGD:
    """Stochastic Gradient Descent (vanilla, handles batched inputs)."""

    def __init__(self, layers, learning_rate=0.01, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.layers:
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * grad_b


class Momentum:
    """SGD with Momentum."""

    def __init__(self, layers, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            self.v_W[i] = self.momentum * self.v_W[i] + grad_W
            self.v_b[i] = self.momentum * self.v_b[i] + grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, layers, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            self.v_W[i] = self.momentum * self.v_W[i] + grad_W
            self.v_b[i] = self.momentum * self.v_b[i] + grad_b
            layer.W -= self.lr * (self.momentum * self.v_W[i] + grad_W)
            layer.b -= self.lr * (self.momentum * self.v_b[i] + grad_b)


class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, layers, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * grad_W ** 2
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * grad_b ** 2
            layer.W -= self.lr * grad_W / (np.sqrt(self.v_W[i]) + self.epsilon)
            layer.b -= self.lr * grad_b / (np.sqrt(self.v_b[i]) + self.epsilon)


class Adam:
    """Adam optimizer."""

    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]
        self.t = 0

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class Nadam:
    """Nadam optimizer (Adam + Nesterov momentum)."""

    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]
        self.t = 0

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Nesterov-style update
            m_W_nesterov = self.beta1 * m_W_hat + (1 - self.beta1) * grad_W / (1 - self.beta1 ** self.t)
            m_b_nesterov = self.beta1 * m_b_hat + (1 - self.beta1) * grad_b / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * m_W_nesterov / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.epsilon)


def get_optimizer(name, layers, learning_rate=0.01, weight_decay=0.0):
    """Factory function to get optimizer by name."""
    optimizers = {
        "sgd": lambda: SGD(layers, learning_rate=learning_rate, weight_decay=weight_decay),
        "momentum": lambda: Momentum(layers, learning_rate=learning_rate, weight_decay=weight_decay),
        "nag": lambda: NAG(layers, learning_rate=learning_rate, weight_decay=weight_decay),
        "rmsprop": lambda: RMSProp(layers, learning_rate=learning_rate, weight_decay=weight_decay),
        "adam": lambda: Adam(layers, learning_rate=learning_rate, weight_decay=weight_decay),
        "nadam": lambda: Nadam(layers, learning_rate=learning_rate, weight_decay=weight_decay),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    return optimizers[name]()
