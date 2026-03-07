"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-mp', '--model_path', type=str, default='best_model.npy',
                        help='Path to saved model weights (default: best_model.npy)')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to evaluate on (default: fashion_mnist)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function (default: cross_entropy)')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer (default: adam)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Epochs (default: 10)')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128],
                        help='Hidden layer sizes (default: [128])')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function (default: relu)')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization (default: xavier)')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='ns26z140-da6401-assignment',
                        help='W&B project name (default: ns26z140-da6401-assignment)')

    return parser.parse_args()


def load_model(model_path):
    """Load trained model from disk."""
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns:
        Dictionary with logits, loss, accuracy, f1, precision, recall
    """
    # Get logits
    logits = model.forward(X_test)

    # Compute loss
    y_test_oh = one_hot_encode(y_test)
    loss = model.compute_loss(logits, y_test_oh)

    # Get predictions
    predictions = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    results = {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }

    return results


def main():
    """Main inference function."""
    args = parse_arguments()

    # Load data
    print(f"Loading {args.dataset} dataset...")
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    # Build model with same architecture
    model = NeuralNetwork(args)

    # Load saved weights
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    print(f"Loading model from {model_path}...")
    weights = load_model(model_path)
    model.set_weights(weights)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print(f"\nEvaluation Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Loss:      {results['loss']:.4f}")

    print("\nEvaluation complete!")
    return results


if __name__ == '__main__':
    main()
