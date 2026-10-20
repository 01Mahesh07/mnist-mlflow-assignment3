"""Assignment 3: four MNIST classification experiments tracked remotely on DagsHub.

Requires a DagsHub account and matching repository; see README.md.
All reported metrics are calculated by actual Keras training, never hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
SEED = 42
EXPERIMENT_NAME = "MNIST_Deep_Learning_Assignment_3"

# Baseline configuration exactly matches the assignment (50 epochs, Adam, 128, 64).
CONFIGS = [
    dict(key="baseline", name="baseline_1x128_adam_bs64_ep50", hidden=[128], optimizer="Adam", batch_size=64, epochs=50),
    dict(key="wider", name="wider_1x256_adam_bs64_ep30", hidden=[256], optimizer="Adam", batch_size=64, epochs=30),
    dict(key="deeper", name="deeper_2x128-64_adam_bs64_ep30", hidden=[128, 64], optimizer="Adam", batch_size=64, epochs=30),
    dict(key="sgd", name="sgd_1x128_bs64_ep30", hidden=[128], optimizer="SGD", batch_size=64, epochs=30),
]


def valid_value(value: str | None, label: str) -> str:
    if not value or value.startswith("YOUR_"):
        raise SystemExit(f"Set {label} in your .env file. See .env.example and README.md.")
    return value.strip()


def split_mnist(tf, np):
    """Fixed 50k training / 10k validation split of MNIST's training subset."""
    (images, labels), (_test_images, _test_labels) = tf.keras.datasets.mnist.load_data()
    images = images.astype("float32") / 255.0  # shape: (60000, 28, 28)
    labels = labels.astype("int64")
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(images))
    train_idx, val_idx = indices[:50_000], indices[50_000:]
    # Do NOT use the official test set for model selection; avoid test leakage.
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def build_model(tf, cfg):
    layers = [tf.keras.Input(shape=(28, 28)), tf.keras.layers.Flatten()]
    for units in cfg["hidden"]:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    model = tf.keras.Sequential(layers, name=cfg["key"] + "_mnist")
    if cfg["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam()
    elif cfg["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD()
    else:
        raise ValueError("Unknown optimizer")
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def prepare_tracking(mlflow, dagshub):
    owner = valid_value(os.getenv("DAGSHUB_OWNER"), "DAGSHUB_OWNER")
    repo = valid_value(os.getenv("DAGSHUB_REPO"), "DAGSHUB_REPO")
    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    actual_uri = mlflow.get_tracking_uri().rstrip("/")
    if actual_uri.rstrip("/") != tracking_uri:
        # Be explicit rather than quietly logging to a local or stale server.
        mlflow.set_tracking_uri(tracking_uri)
    print(f"DagsHub project: https://dagshub.com/{owner}/{repo}")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # The task explicitly requires TensorFlow/Keras autologging. Enable it before fit().
    mlflow.tensorflow.autolog(log_models=True, log_every_epoch=True)
    return owner, repo, tracking_uri


def run_experiment(tf, mlflow, cfg, data, *, smoke=False):
    x_train, y_train, x_val, y_val = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    name = ("smoke_DO_NOT_SUBMIT_" if smoke else "") + cfg["name"]
    epochs = 1 if smoke else cfg["epochs"]
    if smoke:
        x_train, y_train = x_train[:2048], y_train[:2048]
        x_val, y_val = x_val[:512], y_val[:512]
    model = build_model(tf, cfg)
    print(f"\n--- Training {name} ---")
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"dataset": "MNIST", "course_assignment": "Assignment 3", "run_type": "smoke" if smoke else "final"})
        mlflow.log_params({
            "number_of_hidden_layers": len(cfg["hidden"]),
            "nodes_in_each_hidden_layer": ", ".join(str(n) for n in cfg["hidden"]),
            "optimizer": cfg["optimizer"],
            "batch_size": cfg["batch_size"],
            "epochs": epochs,
            "seed": SEED,
            "train_examples": len(x_train),
            "validation_examples": len(x_val),
        })
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=cfg["batch_size"],
            verbose=2,
        )
        values = history.history
        train_acc = float(values["accuracy"][-1]); val_acc = float(values["val_accuracy"][-1])
        train_loss = float(values["loss"][-1]); val_loss = float(values["val_loss"][-1])
        mlflow.log_metrics({
            "final_train_accuracy": train_acc,
            "final_val_accuracy": val_acc,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_epoch_val_accuracy": max(float(v) for v in values["val_accuracy"]),
        })
        row = {
            "run_key": cfg["key"], "run_name": name, "run_id": run.info.run_id,
            "hidden_layers": len(cfg["hidden"]),
            "hidden_nodes": "-".join(str(n) for n in cfg["hidden"]),
            "optimizer": cfg["optimizer"], "batch_size": cfg["batch_size"], "epochs": epochs,
            "train_accuracy": train_acc, "validation_accuracy": val_acc,
            "train_loss": train_loss, "validation_loss": val_loss,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    tf.keras.backend.clear_session()
    return row


def write_rows(path, new_rows):
    # Store latest successful run by configuration key, preserving prior successful
    # configs when --only is used. Do not mix smoke tests into this final summary.
    rows = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["run_key"]] = row
    for r in new_rows:
        rows[r["run_key"]] = r
    ordered = [rows[c["key"]] for c in CONFIGS if c["key"] in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_key", "run_name", "run_id", "hidden_layers", "hidden_nodes", "optimizer",
            "batch_size", "epochs", "train_accuracy", "validation_accuracy", "train_loss",
            "validation_loss", "completed_at_utc"
        ])
        writer.writeheader(); writer.writerows(ordered)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Track MNIST Keras experiments on DagsHub")
    parser.add_argument("--only", choices=["all"] + [c["key"] for c in CONFIGS], default="all",
                        help="Run all four configs or one config (for recovery)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One-epoch connectivity test; NOT valid as submission experiment")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    import dagshub
    import mlflow

    OUTPUT.mkdir(exist_ok=True)
    _, _, uri = prepare_tracking(mlflow, dagshub)
    data = split_mnist(tf, np)
    targets = CONFIGS if args.only == "all" else [c for c in CONFIGS if c["key"] == args.only]
    if args.smoke_test:
        targets = targets[:1]
    all_rows = []
    for cfg in targets:
        row = run_experiment(tf, mlflow, cfg, data, smoke=args.smoke_test)
        all_rows.append(row)
        path = OUTPUT / ("smoke_results.csv" if args.smoke_test else "experiment_results.csv")
        recorded = write_rows(path, [row])
        print(f"Logged to DagsHub: {row['run_name']} (val accuracy {row['validation_accuracy']:.4f})")
        print(f"Saved actual results: {path}")

    if not args.smoke_test:
        saved_path = OUTPUT / "experiment_results.csv"
        with saved_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == 4:
            # Compare final-epoch validation accuracy; lower final validation loss breaks ties.
            best = sorted(rows, key=lambda r: (-float(r["validation_accuracy"]), float(r["validation_loss"])))[0]
            summary = {
                "selection_rule": "highest final-epoch validation accuracy; lowest final-epoch validation loss as tie-breaker",
                "best_run_name": best["run_name"],
                "best_run_id": best["run_id"],
                "best_validation_accuracy": float(best["validation_accuracy"]),
                "best_validation_loss": float(best["validation_loss"]),
                "mlflow_tracking_uri": uri,
                "n_final_configs": 4,
            }
            (OUTPUT / "best_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("\nBEST RUN (by final validation performance):", best["run_name"])
            print("Final validation accuracy:", f"{float(best['validation_accuracy']):.4%}")
        else:
            print(f"\n{len(rows)}/4 final experiments recorded. Run missing configs before submitting.")
    print("\nNext: capture four screenshots in DagsHub and run: python build_report.py")


if __name__ == "__main__":
    main()
