"""
Find the best run from a W&B sweep and print its config.
Usage: python find_best_sweep.py --project ns26z140-da6401-assignment
"""

import argparse
import wandb
import json


def main():
    parser = argparse.ArgumentParser(description="Find best sweep run")
    parser.add_argument("--project", type=str, default="ns26z140-da6401-assignment")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/username (auto-detected if not set)")
    parser.add_argument("--metric", type=str, default="val_accuracy", help="Metric to rank by (default: val_accuracy)")
    args = parser.parse_args()

    api = wandb.Api()

    # Get entity if not provided
    entity = args.entity or api.default_entity

    print(f"Fetching runs from {entity}/{args.project}...")
    all_runs = api.runs(f"{entity}/{args.project}", order=f"-summary_metrics.{args.metric}")

    # Filter to only sweep runs (they have a non-empty config with hyperparameters)
    sweep_runs = []
    for run in all_runs:
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        if config and "optimizer" in config:
            sweep_runs.append(run)

    if not sweep_runs:
        print("No sweep runs found! Run 'python sweep.py --count 100' first.")
        print(f"\nFound {len(list(all_runs))} total runs, but none have sweep configs.")
        return

    print(f"Found {len(sweep_runs)} sweep runs (filtered from {len(list(all_runs))} total).")
    best_run = sweep_runs[0]
    print(f"\n{'='*60}")
    print(f"BEST RUN: {best_run.name}")
    print(f"Run ID:   {best_run.id}")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    for key in ["val_accuracy", "test_accuracy", "test_f1", "train_accuracy", "train_loss"]:
        val = best_run.summary.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    print(f"\nBest Config:")
    config = {k: v for k, v in best_run.config.items() if not k.startswith("_")}
    print(json.dumps(config, indent=2))

    # Print the train.py command to reproduce
    print(f"\nCommand to train with best config:")
    cmd = "python train.py"
    mapping = {
        "dataset": "-d", "epochs": "-e", "batch_size": "-b",
        "loss": "-l", "optimizer": "-o", "learning_rate": "-lr",
        "weight_decay": "-wd", "num_layers": "-nhl",
        "hidden_size": "-sz", "activation": "-a", "weight_init": "-w_i",
    }
    for key, flag in mapping.items():
        if key in config:
            val = config[key]
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            cmd += f" {flag} {val}"
    cmd += f" -w_p {args.project}"
    print(cmd)

    # Also show top 5
    print(f"\nTop 5 sweep runs by {args.metric}:")
    for i, run in enumerate(sweep_runs[:5]):
        val = run.summary.get(args.metric, "N/A")
        f1 = run.summary.get("test_f1", "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        print(f"  {i+1}. {run.name} | {args.metric}={val} | test_f1={f1}")


if __name__ == "__main__":
    main()
