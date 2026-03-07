# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

A configurable, modular Multi-Layer Perceptron (MLP) built from scratch using only **NumPy** for mathematical operations. Implements the complete training pipeline — forward propagation, backpropagation, various optimizers, activation functions, and loss functions — to classify MNIST and Fashion-MNIST datasets.

- **GitHub Repository**: `<YOUR_GITHUB_LINK>`

- **W&B Report**: https://wandb.ai/ns26z140-iitm-india/ns26z140-da6401-assignment/reports/NS26Z140-ASSIGNEMENT-1---VmlldzoxNjEyODkyMw?accessToken=vj39lsx5irq5miwv092yjq2bh4n6mmf60y5azhbs7397dv4oc05skbkzuh0sq3mf

## Project Structure

```
.
├── README.md
├── requirements.txt
├── models/
├── notebooks/
└── src/
    ├── ann/
    │   ├── __init__.py
    │   ├── activations.py        # Sigmoid, Tanh, ReLU, Softmax + derivatives
    │   ├── neural_layer.py       # Single layer: forward, backward, weight init
    │   ├── neural_network.py     # Full MLP: forward/backward/train/evaluate
    │   ├── objective_functions.py # Cross-Entropy, MSE + derivatives
    │   └── optimizers.py         # SGD, Momentum, NAG, RMSProp, Adam, Nadam
    ├── utils/
    │   ├── __init__.py
    │   └── data_loader.py        # MNIST/Fashion-MNIST loading & preprocessing
    ├── train.py                  # Training script with CLI
    ├── inference.py              # Inference & metrics script
    ├── sweep.py                  # W&B hyperparameter sweep (100+ runs)
    ├── wandb_experiments.py      # All W&B report experiments (Sections 2.1-2.10)
    ├── best_model.npy            # Best model weights (generated after training)
    └── best_config.json          # Best hyperparameter configuration
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
cd src
python train.py -d fashion_mnist -e 10 -b 64 -l cross_entropy -o adam -lr 0.001 -nhl 3 -sz 128 -a relu -w_i xavier -w_p ns26z140-da6401-assignment
```

### CLI Arguments

| Flag     | Long                | Description                                                      | Default                        |
| -------- | ------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `-d`   | `--dataset`       | `mnist` or `fashion_mnist`                                   | `fashion_mnist`              |
| `-e`   | `--epochs`        | Number of training epochs                                        | `10`                         |
| `-b`   | `--batch_size`    | Mini-batch size                                                  | `64`                         |
| `-l`   | `--loss`          | `cross_entropy` or `mean_squared_error`                      | `cross_entropy`              |
| `-o`   | `--optimizer`     | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `adam`                       |
| `-lr`  | `--learning_rate` | Initial learning rate                                            | `0.001`                      |
| `-wd`  | `--weight_decay`  | L2 regularization weight decay                                   | `0.0`                        |
| `-nhl` | `--num_layers`    | Number of hidden layers                                          | `3`                          |
| `-sz`  | `--hidden_size`   | Neurons per hidden layer (list)                                  | `[128]`                      |
| `-a`   | `--activation`    | `sigmoid`, `tanh`, `relu`                                  | `relu`                       |
| `-w_i` | `--weight_init`   | `random` or `xavier`                                         | `xavier`                     |
| `-w_p` | `--wandb_project` | W&B project name                                                 | `ns26z140-da6401-assignment` |

## Inference

```bash
cd src
python inference.py -d fashion_mnist -nhl 3 -sz 128 -a relu -mp best_model.npy
```

Outputs: **Accuracy, Precision, Recall, F1-Score**

## W&B Hyperparameter Sweep (Section 2.2)

```bash
cd src
python sweep.py --project ns26z140-da6401-assignment --count 100
```

## W&B Experiments (Sections 2.1-2.10)

Run all experiments:

```bash
cd src
python wandb_experiments.py --section all --project ns26z140-da6401-assignment
```

Run a specific section:

```bash
python wandb_experiments.py --section 2.1
python wandb_experiments.py --section 2.3
python wandb_experiments.py --section 2.9
```

## Contact

For questions or issues, please contact the teaching staff or post on the course forum.
