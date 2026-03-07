"""
W&B Hyperparameter Sweep Script
Performs at least 100 runs with varying hyperparameters for Section 2.2
"""

import sys
import os
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode
from sklearn.metrics import f1_score


# Sweep configuration
sweep_config = {
    "method": "bayes",
    "name": "da6401-sweep",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "weight_decay": {"values": [0, 0.0005, 0.005]},
        "loss": {"values": ["cross_entropy", "mean_squared_error"]},
    }
}


def train_sweep():
    """Single sweep run."""
    run = wandb.init()
    config = wandb.config

    # Build argument namespace
    class Args:
        pass

    args = Args()
    args.dataset = "fashion_mnist"
    args.epochs = config.epochs
    args.batch_size = config.batch_size
    args.loss = config.loss
    args.optimizer = config.optimizer
    args.learning_rate = config.learning_rate
    args.weight_decay = config.weight_decay
    args.num_layers = config.num_layers
    args.hidden_size = [config.hidden_size]
    args.activation = config.activation
    args.weight_init = config.weight_init

    # Set run name
    run.name = (f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}_"
                f"opt_{args.optimizer}_lr_{args.learning_rate}_wd_{args.weight_decay}_"
                f"wi_{args.weight_init}_loss_{args.loss}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)

    # Build model
    model = NeuralNetwork(args)

    num_samples = X_train.shape[0]
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            loss = model.train_step(X_shuffled[start:end], y_shuffled[start:end])
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_acc = model.evaluate(X_train, y_train)
        val_acc = model.evaluate(X_val, y_val)

        val_logits = model.forward(X_val)
        val_loss = model.compute_loss(val_logits, y_val_oh)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Test evaluation
    test_acc = model.evaluate(X_test, y_test)
    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average='macro')

    wandb.log({
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "best_val_accuracy": best_val_acc,
    })

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ns26z140-da6401-assignment")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep runs")
    cli_args = parser.parse_args()

    sweep_id = wandb.sweep(sweep_config, project=cli_args.project)
    wandb.agent(sweep_id, function=train_sweep, count=cli_args.count)
