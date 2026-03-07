"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import sys
import os
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Mini-batch size (default: 64)')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function (default: cross_entropy)')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer (default: adam)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='Weight decay for L2 regularization (default: 0.0)')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128],
                        help='Number of neurons in each hidden layer (default: [128])')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function (default: relu)')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method (default: xavier)')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='ns26z140-da6401-assignment',
                        help='W&B project name (default: ns26z140-da6401-assignment)')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    # Load data
    print(f"Loading {args.dataset} dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    # One-hot encode labels
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    # Initialize W&B
    run_name = f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}_opt_{args.optimizer}_lr_{args.learning_rate}_wd_{args.weight_decay}_wi_{args.weight_init}_loss_{args.loss}"
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "weight_init": args.weight_init,
            }
        )

    # Build model
    model = NeuralNetwork(args)
    print(f"Model built: {args.num_layers} hidden layers, activation={args.activation}, "
          f"optimizer={args.optimizer}, lr={args.learning_rate}")

    # Training loop
    num_samples = X_train.shape[0]
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(args.epochs):
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_oh_shuffled = y_train_oh[indices]

        # Mini-batch training
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_oh_shuffled[start:end]

            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate on training and validation sets
        train_acc = model.evaluate(X_train, y_train)
        val_acc = model.evaluate(X_val, y_val)

        # Compute validation loss
        val_logits = model.forward(X_val)
        val_loss = model.compute_loss(val_logits, y_val_oh)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log to W&B
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    # Evaluate on test set with best model
    if best_weights is not None:
        model.set_weights(best_weights)

    test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Compute test F1 score
    from sklearn.metrics import f1_score, precision_score, recall_score
    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average='macro')
    test_precision = precision_score(y_test, test_preds, average='macro')
    test_recall = recall_score(y_test, test_preds, average='macro')

    print(f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    if WANDB_AVAILABLE:
        wandb.log({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
        })

    # Save best model weights
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    best_model_path = os.path.join(save_dir, "best_model.npy")
    best_config_path = os.path.join(save_dir, "best_config.json")

    np.save(best_model_path, model.get_weights())
    print(f"Best model saved to {best_model_path}")

    # Save best config
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
    }
    with open(best_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Best config saved to {best_config_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()
