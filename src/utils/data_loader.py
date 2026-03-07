"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
import os
import gzip
import struct
import urllib.request


def _download_file(url, filepath):
    """Download a file from a URL."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def _read_idx_images(filepath):
    """Read IDX image file format."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(filepath):
    """Read IDX label file format."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def _load_dataset_manual(dataset_name):
    """Download and load MNIST or Fashion-MNIST manually."""
    if dataset_name == "mnist":
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
    elif dataset_name == "fashion_mnist":
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mlp_data", dataset_name)

    paths = {}
    for key, fname in files.items():
        paths[key] = _download_file(base_url + fname, os.path.join(cache_dir, fname))

    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    return (X_train, y_train), (X_test, y_test)


def _load_dataset(dataset_name):
    """Load dataset using keras if available, else download manually."""
    try:
        from keras.datasets import mnist, fashion_mnist
        if dataset_name == "mnist":
            return mnist.load_data()
        elif dataset_name == "fashion_mnist":
            return fashion_mnist.load_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist
        if dataset_name == "mnist":
            return mnist.load_data()
        elif dataset_name == "fashion_mnist":
            return fashion_mnist.load_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    # Fallback: download manually
    return _load_dataset_manual(dataset_name)


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
    (X_train_full, y_train_full), (X_test, y_test) = _load_dataset(dataset_name)

    # Flatten images: (N, 28, 28) -> (N, 784)
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    # Split training into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split, random_state=42, stratify=y_train_full
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot
