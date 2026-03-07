"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name="mnist", val_split=0.1):
    """
    Load and preprocess dataset.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        val_split: Fraction of training data to use for validation

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (all numpy arrays)
        X: flattened and normalized (N, 784), values in [0, 1]
        y: integer labels (N,)
    """
    if dataset_name == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'mnist' or 'fashion_mnist'")

    # Flatten images: (N, 28, 28) -> (N, 784)
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split, random_state=42, stratify=y_train_full
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot
