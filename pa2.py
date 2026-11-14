import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def evaluate(model, inputs, labels, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = loss_fn(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Compare the four Part A activations.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("part_a_results"))
    output_dir = parser.parse_args().output_dir

    data_dir = Path(__file__).resolve().parent / "supporting_files"
    training = pd.read_csv(data_dir / "training.csv")
    validation = pd.read_csv(data_dir / "validation.csv")

    feature_columns = [f"x{i}" for i in range(1, 25)]
    x_train = training[feature_columns].to_numpy(dtype=np.float64)
    y_train = training["condition"].to_numpy(dtype=np.int64)
    x_val = validation[feature_columns].to_numpy(dtype=np.float64)
    y_val = validation["condition"].to_numpy(dtype=np.int64)

    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Both datasets must contain finite sensor readings.")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    if (std <= 0).any():
        raise ValueError("Every training sensor must have positive standard deviation.")

    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std

    print(f"Training inputs: {x_train_scaled.shape}; labels: {y_train.shape}")
    print(f"Validation inputs: {x_val_scaled.shape}; labels: {y_val.shape}")
    print(f"Training classes: {np.unique(y_train).tolist()}")
    print(f"First training x1: {x_train[0, 0]:.4f} -> {x_train_scaled[0, 0]:.4f}")
    print(f"Standardized training x1 mean: {x_train_scaled[:, 0].mean():.2e}")
    print(f"Standardized training x1 population std: {x_train_scaled[:, 0].std(ddof=0):.6f}")

    train_inputs = torch.from_numpy(x_train_scaled.astype(np.float32))
    val_inputs = torch.from_numpy(x_val_scaled.astype(np.float32))
    train_labels = torch.from_numpy(y_train.copy())
    val_labels = torch.from_numpy(y_val.copy())

    training_data = TensorDataset(train_inputs, train_labels)
    activations = {
        "Identity": nn.Identity,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
    }
    results = []
    figure, (accuracy_axis, loss_axis) = plt.subplots(2, 1, sharex=True, layout="constrained")
    for name, activation in activations.items():
        history = train_activation(activation, training_data, val_inputs, val_labels)
        results.extend({"activation": name, **row} for row in history)
        epochs = [row["epoch"] for row in history]
        accuracy_axis.plot(epochs, [100 * row["val_accuracy"] for row in history], label=name)
        loss_axis.plot(epochs, [row["val_loss"] for row in history], label=name)
        print(f"{name:8s} | epoch 120 validation accuracy {history[-1]['val_accuracy']:.2%}")

    accuracy_axis.set_ylabel("Validation accuracy (%)")
    loss_axis.set_ylabel("Validation cross-entropy")
    loss_axis.set_xlabel("Epoch")
    for axis in (accuracy_axis, loss_axis):
        axis.legend()
        axis.grid(alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    figure.savefig(output_dir / "validation_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved Part A results in {output_dir.resolve()}")


def train_activation(activation, training_data, val_inputs, val_labels):
    train_inputs, train_labels = training_data.tensors
    seed = 314159
    torch.manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_batches = DataLoader(
        training_data,
        batch_size=64,
        shuffle=True,
        generator=shuffle_generator,
    )

    model = nn.Sequential(nn.Linear(24, 32), activation(), nn.Linear(32, 6))
    for layer in (model[0], model[2]):
        nn.init.normal_(layer.weight, mean=0.0, std=layer.in_features ** -0.5)
        nn.init.zeros_(layer.bias)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0, weight_decay=0)

    history = []
    for epoch in range(1, 121):
        model.train()
        for inputs, labels in train_batches:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, train_inputs, train_labels, loss_fn)
        val_loss, val_accuracy = evaluate(model, val_inputs, val_labels, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
    return history


if __name__ == "__main__":
    main()
