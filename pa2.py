import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Experiment:
    name: str
    activation: type[nn.Module] = nn.ReLU
    widths: tuple[int, ...] = (32,)
    initialization: str = "kaiming"
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    quadratic: bool = False


def part_a():
    return [
        Experiment(name, activation=activation, initialization="nielsen")
        for name, activation in (
            ("Identity", nn.Identity), ("Sigmoid", nn.Sigmoid),
            ("Tanh", nn.Tanh), ("ReLU", nn.ReLU),
        )
    ]


def part_b():
    return [
        Experiment("Cross-entropy"),
        Experiment("Quadratic cost", quadratic=True),
    ]


def quadratic_cost(logits, labels):
    probabilities = logits.softmax(dim=1)
    targets = nn.functional.one_hot(labels, num_classes=logits.shape[1])
    return 0.5 * (probabilities - targets).square().sum(dim=1).mean()


def part_c():
    return [
        Experiment(name, widths=(64, 32), initialization=method, learning_rate=0.03)
        for name, method in (
            ("Small normal", "small"), ("Large normal", "large"),
            ("Nielsen", "nielsen"), ("Kaiming", "kaiming"),
        )
    ]


def part_d():
    return [
        Experiment(name, widths=widths, learning_rate=0.03)
        for name, widths in (
            ("Tiny", (8,)), ("Small", (32,)), ("Medium", (64, 32)), ("Large", (128, 128)),
        )
    ]


def part_e():
    return [
        Experiment(f"{decay:g}", widths=(128, 128), learning_rate=0.03, weight_decay=decay)
        for decay in (0, 0.001, 0.01, 0.03, 0.1)
    ]


def evaluate(model, inputs, labels, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = loss_fn(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Run controlled PA2 experiments.")
    parser.add_argument("output_dir", nargs="?", type=Path)
    parts = {"a": part_a, "b": part_b, "c": part_c, "d": part_d, "e": part_e}
    parser.add_argument("--part", choices=parts, default="a")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"part_{args.part}_results")

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
    experiments = parts[args.part]()
    histories = {}
    final_metrics = []
    for experiment in experiments:
        history, model = train_experiment(experiment, training_data, val_inputs, val_labels)
        histories[experiment.name] = history
        final_row = {"configuration": experiment.name, **history[-1]}
        if args.part == "e":
            final_row["weight_l2_norm"] = weight_l2_norm(model)
        final_metrics.append(final_row)
        best = max(history, key=lambda row: row["val_accuracy"])
        print(
            f"{experiment.name} | final validation accuracy {history[-1]['val_accuracy']:.2%}"
            f" | peak {best['val_accuracy']:.2%} at epoch {best['epoch']}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "metrics.csv",
        [{"configuration": name, **row} for name, history in histories.items() for row in history],
    )
    if args.part in ("d", "e"):
        write_csv(output_dir / "final_metrics.csv", final_metrics)
    plot_results(args.part, histories, output_dir)
    print(f"Saved Part {args.part.upper()} results in {output_dir.resolve()}")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path, panels):
    figure, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False, layout="constrained")
    for axis, (ylabel, series) in zip(axes[:, 0], panels):
        for name, history, metric in series:
            values = [row[metric] for row in history]
            if metric.endswith("accuracy"):
                values = [100 * value for value in values]
            axis.plot([row["epoch"] for row in history], values, label=name)
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Epoch")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_results(part, histories, output_dir):
    accuracy = ("Validation accuracy (%)", [(name, rows, "val_accuracy") for name, rows in histories.items()])
    loss = ("Validation cross-entropy", [(name, rows, "val_loss") for name, rows in histories.items()])
    if part == "b":
        plot_metrics(output_dir / "validation_accuracy.png", [accuracy])
        for name, filename in (
            ("Cross-entropy", "cross_entropy_loss.png"), ("Quadratic cost", "quadratic_loss.png")
        ):
            plot_metrics(output_dir / filename, [(f"Validation {name.lower()}", [(name, histories[name], "val_loss")])])
    elif part == "d":
        plot_metrics(output_dir / "validation_accuracy.png", [accuracy])
        large = histories["Large"]
        plot_metrics(
            output_dir / "large_training_and_validation.png",
            [
                ("Accuracy (%)", [("Training", large, "train_accuracy"), ("Validation", large, "val_accuracy")]),
                ("Cross-entropy", [("Training", large, "train_loss"), ("Validation", large, "val_loss")]),
            ],
        )
    else:
        plot_metrics(output_dir / "validation_curves.png", [accuracy, loss])


def build_model(experiment):
    sizes = (24, *experiment.widths, 6)
    modules = []
    for layer_number, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        modules.append(nn.Linear(inputs, outputs))
        if layer_number < len(experiment.widths):
            modules.append(experiment.activation())

    for layer in modules[::2]:
        if experiment.initialization == "kaiming":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif experiment.initialization == "nielsen":
            nn.init.normal_(layer.weight, mean=0.0, std=layer.in_features ** -0.5)
        else:
            std = {"small": 0.01, "large": 1.0}[experiment.initialization]
            nn.init.normal_(layer.weight, mean=0.0, std=std)
        nn.init.zeros_(layer.bias)
    return nn.Sequential(*modules)


def weight_l2_norm(model):
    with torch.no_grad():
        weights = torch.cat([layer.weight.flatten() for layer in model if isinstance(layer, nn.Linear)])
        return weights.norm().item()


def train_experiment(experiment, training_data, val_inputs, val_labels):
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

    model = build_model(experiment)
    loss_fn = quadratic_cost if experiment.quadratic else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=experiment.learning_rate, momentum=0, weight_decay=experiment.weight_decay
    )

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
    return history, model


if __name__ == "__main__":
    main()
