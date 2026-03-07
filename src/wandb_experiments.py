"""
W&B Experiments Script
Runs all experiments required for the W&B report (Sections 2.1 - 2.10)
"""

import sys
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import json


WANDB_PROJECT = "ns26z140-da6401-assignment"


def make_args(**kwargs):
    """Create an argument namespace from keyword arguments."""
    class Args:
        pass
    args = Args()
    defaults = {
        "dataset": "mnist",
        "epochs": 10,
        "batch_size": 64,
        "loss": "cross_entropy",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "num_layers": 3,
        "hidden_size": [128],
        "activation": "relu",
        "weight_init": "xavier",
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def train_model(args, log_gradients=False, log_activations=False, return_history=False):
    """Generic training function with optional gradient/activation logging."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)

    model = NeuralNetwork(args)
    num_samples = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    gradient_history = []
    activation_history = []

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

            # Log gradients for first few iterations
            if log_gradients and epoch == 0 and num_batches <= 50:
                # Gradient norm of first hidden layer
                grad_norm = np.linalg.norm(model.layers[0].grad_W)
                gradient_history.append({
                    "iteration": (epoch * (num_samples // args.batch_size)) + num_batches,
                    "grad_norm": grad_norm,
                    "grad_W": model.layers[0].grad_W.copy(),
                })

            if log_activations and num_batches == 1:
                acts = []
                for layer in model.layers[:-1]:
                    acts.append(layer.output.copy())
                activation_history.append(acts)

        avg_loss = epoch_loss / num_batches
        train_acc = model.evaluate(X_train, y_train)
        val_acc = model.evaluate(X_val, y_val)
        val_logits = model.forward(X_val)
        val_loss = model.compute_loss(val_logits, y_val_oh)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

    test_acc = model.evaluate(X_test, y_test)
    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average='macro')

    wandb.log({"test_accuracy": test_acc, "test_f1": test_f1})

    result = {
        "model": model, "history": history,
        "test_acc": test_acc, "test_f1": test_f1,
        "test_preds": test_preds, "y_test": y_test,
        "X_test": X_test,
    }
    if log_gradients:
        result["gradient_history"] = gradient_history
    if log_activations:
        result["activation_history"] = activation_history
    return result


# =========================================================================
# Section 2.1: Data Exploration and Class Distribution
# =========================================================================
def experiment_2_1_data_exploration():
    """Log W&B Table with 5 sample images from each class."""
    print("=== Section 2.1: Data Exploration ===")
    from keras.datasets import fashion_mnist
    (X_train, y_train), _ = fashion_mnist.load_data()

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    run = wandb.init(project=WANDB_PROJECT, name="2.1_data_exploration", reinit=True)

    # Create a W&B Table
    columns = ["Class", "Class Name", "Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]
    table = wandb.Table(columns=columns)

    for cls in range(10):
        cls_indices = np.where(y_train == cls)[0]
        sample_indices = np.random.choice(cls_indices, size=5, replace=False)
        images = [wandb.Image(X_train[idx]) for idx in sample_indices]
        table.add_data(cls, class_names[cls], *images)

    wandb.log({"sample_images": table})

    # Also log class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Fashion-MNIST Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    wandb.log({"class_distribution": wandb.Image(fig)})
    plt.close()

    wandb.finish()
    print("Section 2.1 complete.")


# =========================================================================
# Section 2.3: Optimizer Showdown
# =========================================================================
def experiment_2_3_optimizer_showdown():
    """Compare 4 optimizers with same architecture."""
    print("=== Section 2.3: Optimizer Showdown ===")
    optimizers = ["sgd", "momentum", "nag", "rmsprop"]

    for opt in optimizers:
        run = wandb.init(project=WANDB_PROJECT, name=f"2.3_optimizer_{opt}", reinit=True)
        args = make_args(
            dataset="mnist", epochs=5, num_layers=3, hidden_size=[128],
            activation="relu", optimizer=opt, learning_rate=0.001,
            weight_init="xavier", loss="cross_entropy"
        )
        result = train_model(args)
        wandb.finish()

    print("Section 2.3 complete.")


# =========================================================================
# Section 2.4: Vanishing Gradient Analysis
# =========================================================================
def experiment_2_4_vanishing_gradient():
    """Compare Sigmoid vs ReLU gradient norms with RMSProp."""
    print("=== Section 2.4: Vanishing Gradient Analysis ===")

    for activation in ["sigmoid", "relu"]:
        run = wandb.init(project=WANDB_PROJECT, name=f"2.4_gradient_{activation}", reinit=True)
        args = make_args(
            dataset="mnist", epochs=5, num_layers=5, hidden_size=[128],
            activation=activation, optimizer="rmsprop", learning_rate=0.001,
            weight_init="xavier", loss="cross_entropy"
        )

        X_train, y_train, X_val, y_val, _, _ = load_data(args.dataset)
        y_train_oh = one_hot_encode(y_train)
        model = NeuralNetwork(args)
        num_samples = X_train.shape[0]

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

                # Log gradient norms for first hidden layer
                grad_norm = np.linalg.norm(model.layers[0].grad_W)
                step = epoch * (num_samples // args.batch_size) + num_batches
                wandb.log({"gradient_norm_layer0": grad_norm, "step": step})

            avg_loss = epoch_loss / num_batches
            train_acc = model.evaluate(X_train, y_train)
            val_acc = model.evaluate(X_val, y_val)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            })

        wandb.finish()

    print("Section 2.4 complete.")


# =========================================================================
# Section 2.5: Dead Neuron Investigation
# =========================================================================
def experiment_2_5_dead_neurons():
    """Monitor activations with ReLU (high lr) vs Tanh."""
    print("=== Section 2.5: Dead Neuron Investigation ===")

    configs = [
        ("relu", 0.1, "2.5_dead_neuron_relu_highlr"),
        ("tanh", 0.1, "2.5_dead_neuron_tanh_highlr"),
    ]

    for activation, lr, name in configs:
        run = wandb.init(project=WANDB_PROJECT, name=name, reinit=True)
        args = make_args(
            dataset="mnist", epochs=10, num_layers=3, hidden_size=[128],
            activation=activation, optimizer="sgd", learning_rate=lr,
            weight_init="xavier", loss="cross_entropy"
        )

        X_train, y_train, X_val, y_val, _, _ = load_data(args.dataset)
        y_train_oh = one_hot_encode(y_train)
        model = NeuralNetwork(args)
        num_samples = X_train.shape[0]

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

            # Log activation statistics
            sample = X_train[:1000]
            model.forward(sample)
            for layer_idx, layer in enumerate(model.layers[:-1]):
                acts = layer.output
                zero_frac = np.mean(acts == 0)
                mean_act = np.mean(acts)
                wandb.log({
                    f"layer{layer_idx}_zero_fraction": zero_frac,
                    f"layer{layer_idx}_mean_activation": mean_act,
                    "epoch": epoch + 1,
                })

            avg_loss = epoch_loss / num_batches
            train_acc = model.evaluate(X_train, y_train)
            val_acc = model.evaluate(X_val, y_val)

            wandb.log({
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            })

        wandb.finish()

    print("Section 2.5 complete.")


# =========================================================================
# Section 2.6: Loss Function Comparison
# =========================================================================
def experiment_2_6_loss_comparison():
    """Compare MSE vs Cross-Entropy."""
    print("=== Section 2.6: Loss Function Comparison ===")

    for loss_fn in ["cross_entropy", "mean_squared_error"]:
        run = wandb.init(project=WANDB_PROJECT, name=f"2.6_loss_{loss_fn}", reinit=True)
        args = make_args(
            dataset="mnist", epochs=10, num_layers=3, hidden_size=[128],
            activation="relu", optimizer="adam", learning_rate=0.001,
            weight_init="xavier", loss=loss_fn
        )
        result = train_model(args)
        wandb.finish()

    print("Section 2.6 complete.")


# =========================================================================
# Section 2.8: Error Analysis (Confusion Matrix)
# =========================================================================
def experiment_2_8_confusion_matrix():
    """Plot confusion matrix for best model."""
    print("=== Section 2.8: Error Analysis ===")

    run = wandb.init(project=WANDB_PROJECT, name="2.8_confusion_matrix", reinit=True)
    args = make_args(
        dataset="mnist", epochs=10, num_layers=3, hidden_size=[128],
        activation="relu", optimizer="adam", learning_rate=0.001,
        weight_init="xavier", loss="cross_entropy"
    )
    result = train_model(args)

    # Confusion matrix
    cm = confusion_matrix(result["y_test"], result["test_preds"])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')

    # Add text annotations
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

    # Also log as W&B confusion matrix
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(
        probs=None, y_true=result["y_test"], preds=result["test_preds"],
        class_names=[str(i) for i in range(10)]
    )})

    wandb.finish()
    print("Section 2.8 complete.")


# =========================================================================
# Section 2.9: Weight Initialization & Symmetry
# =========================================================================
def experiment_2_9_weight_init():
    """Compare Zeros vs Xavier initialization."""
    print("=== Section 2.9: Weight Initialization ===")

    for init_method, init_name in [("zeros", "zeros"), ("xavier", "xavier")]:
        run = wandb.init(project=WANDB_PROJECT, name=f"2.9_init_{init_name}", reinit=True)
        args = make_args(
            dataset="mnist", epochs=5, num_layers=3, hidden_size=[128],
            activation="relu", optimizer="adam", learning_rate=0.001,
            weight_init=init_method, loss="cross_entropy", batch_size=64
        )

        X_train, y_train, X_val, y_val, _, _ = load_data(args.dataset)
        y_train_oh = one_hot_encode(y_train)
        model = NeuralNetwork(args)
        num_samples = X_train.shape[0]

        neuron_grads = {f"neuron_{j}": [] for j in range(5)}

        iteration = 0
        for epoch in range(args.epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            for start in range(0, num_samples, args.batch_size):
                end = min(start + args.batch_size, num_samples)
                model.train_step(X_shuffled[start:end], y_shuffled[start:end])
                iteration += 1

                if iteration <= 50:
                    # Log gradients for 5 neurons in first hidden layer
                    for j in range(5):
                        grad_val = np.linalg.norm(model.layers[0].grad_W[:, j])
                        neuron_grads[f"neuron_{j}"].append(grad_val)
                        wandb.log({f"grad_neuron_{j}": grad_val, "iteration": iteration})

            train_acc = model.evaluate(X_train, y_train)
            val_acc = model.evaluate(X_val, y_val)
            wandb.log({"epoch": epoch + 1, "train_accuracy": train_acc, "val_accuracy": val_acc})

        # Plot gradient lines
        fig, ax = plt.subplots(figsize=(10, 6))
        for j in range(5):
            ax.plot(range(1, len(neuron_grads[f"neuron_{j}"]) + 1),
                    neuron_grads[f"neuron_{j}"], label=f"Neuron {j}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(f"Gradient Norms ({init_name} init) - First 50 iterations")
        ax.legend()
        plt.tight_layout()
        wandb.log({"gradient_plot": wandb.Image(fig)})
        plt.close()

        wandb.finish()

    print("Section 2.9 complete.")


# =========================================================================
# Section 2.10: Fashion-MNIST Transfer Challenge
# =========================================================================
def experiment_2_10_fashion_mnist():
    """Run 3 configs on Fashion-MNIST."""
    print("=== Section 2.10: Fashion-MNIST Transfer ===")

    configs = [
        {"activation": "relu", "optimizer": "adam", "num_layers": 3, "hidden_size": [128], "learning_rate": 0.001},
        {"activation": "relu", "optimizer": "nadam", "num_layers": 4, "hidden_size": [128], "learning_rate": 0.001},
        {"activation": "tanh", "optimizer": "rmsprop", "num_layers": 3, "hidden_size": [128], "learning_rate": 0.001},
    ]

    for i, cfg in enumerate(configs):
        run = wandb.init(project=WANDB_PROJECT,
                         name=f"2.10_fmnist_config{i+1}_{cfg['activation']}_{cfg['optimizer']}",
                         reinit=True)
        args = make_args(
            dataset="fashion_mnist", epochs=10, weight_init="xavier",
            loss="cross_entropy", batch_size=64, **cfg
        )
        result = train_model(args)
        print(f"  Config {i+1}: Test Acc={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
        wandb.finish()

    print("Section 2.10 complete.")


# =========================================================================
# Main: Run all experiments
# =========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run W&B experiments")
    parser.add_argument("--section", type=str, default="all",
                        help="Section to run (2.1, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9, 2.10, or all)")
    parser.add_argument("--project", type=str, default="ns26z140-da6401-assignment")
    cli = parser.parse_args()

    WANDB_PROJECT = cli.project

    sections = {
        "2.1": experiment_2_1_data_exploration,
        "2.3": experiment_2_3_optimizer_showdown,
        "2.4": experiment_2_4_vanishing_gradient,
        "2.5": experiment_2_5_dead_neurons,
        "2.6": experiment_2_6_loss_comparison,
        "2.8": experiment_2_8_confusion_matrix,
        "2.9": experiment_2_9_weight_init,
        "2.10": experiment_2_10_fashion_mnist,
    }

    if cli.section == "all":
        for name, func in sections.items():
            func()
    elif cli.section in sections:
        sections[cli.section]()
    else:
        print(f"Unknown section: {cli.section}. Available: {list(sections.keys())}")
