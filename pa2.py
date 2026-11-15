import argparse
import csv
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from statistics import fmean

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


def selected_large_model(decay):
    return Experiment(f"{decay:.12g}", widths=(128, 128), learning_rate=0.03, weight_decay=decay)


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
    parser.add_argument("--part", choices=(*parts, "matrix", "search", "final"), default="a")
    parser.add_argument("--decay", type=float, help="Weight decay for matrix mode (default: 0.01).")
    parser.add_argument("--epochs", type=int, help="Epoch count for matrix mode (default: 120).")
    parser.add_argument("--json", type=Path, help="Final JSON path (default: pa2_model.json).")
    args = parser.parse_args()
    if args.part != "matrix" and (args.decay is not None or args.epochs is not None):
        parser.error("--decay and --epochs apply to matrix mode")
    if args.decay is not None and not 0 <= args.decay < float("inf"):
        parser.error("--decay must be finite and nonnegative")
    if args.epochs is not None and not 1 <= args.epochs <= 120:
        parser.error("--epochs must be between 1 and 120")
    if args.json is not None and args.part != "final":
        parser.error("--json applies to final mode")
    if args.part == "final" and args.output_dir is not None:
        parser.error("use --json to specify the final export path")
    default_output = {
        "matrix": "validation_results",
        "search": "decay_search_results",
    }.get(args.part, f"part_{args.part}_results")
    output_dir = args.output_dir or Path(default_output)

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

    if args.part == "final":
        combined = TensorDataset(
            torch.cat((train_inputs, val_inputs)),
            torch.cat((train_labels, val_labels)),
        )
        export_final_model(combined, args.json or Path("pa2_model.json"))
        return

    training_data = TensorDataset(train_inputs, train_labels)
    if args.part == "matrix":
        decay = 0.01 if args.decay is None else args.decay
        experiment = selected_large_model(decay)
        history, model = train_experiment(
            experiment, training_data, val_inputs, val_labels, epochs=args.epochs or 120
        )
        save_confusion_matrix(model, val_inputs, val_labels, output_dir)
        print(f"Epoch {history[-1]['epoch']} validation accuracy {history[-1]['val_accuracy']:.2%}")
        return
    if args.part == "search":
        search_weight_decay(training_data, val_inputs, val_labels, output_dir)
        return

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


def search_weight_decay(training_data, inputs, labels, output_dir):
    histories = []
    trials = []

    def run(decay, lower=None, upper=None, reference=None):
        history, model = train_experiment(selected_large_model(decay), training_data, inputs, labels)
        with torch.no_grad():
            correct = model(inputs).argmax(dim=1) == labels
        recent = history[-10:]
        if reference is None:
            reference = correct
        trial = {
            "trial": len(trials),
            "weight_decay": decay,
            "interval_lower": lower,
            "interval_upper": upper,
            "mean_last_10_val_loss": fmean(row["val_loss"] for row in recent),
            "mean_last_10_val_accuracy": fmean(row["val_accuracy"] for row in recent),
            "epoch_120_val_accuracy": history[-1]["val_accuracy"],
            "epoch_120_val_loss": history[-1]["val_loss"],
            "epoch_120_correct": correct.sum().item(),
            "candidate_only_correct": (correct & ~reference).sum().item(),
            "baseline_only_correct": (reference & ~correct).sum().item(),
        }
        trials.append(trial)
        histories.extend({"trial": trial["trial"], "weight_decay": decay, **row} for row in history)
        print(
            f"Trial {trial['trial']} | decay {decay:.8g} | "
            f"validation accuracy {history[-1]['val_accuracy']:.2%} | "
            f"mean validation loss {trial['mean_last_10_val_loss']:.5f}",
            flush=True,
        )
        return trial["mean_last_10_val_loss"], correct

    baseline_loss, baseline_correct = run(0.01)
    lower, upper = log(0.005), log(0.02)
    ratio = (sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)

    def probe(point):
        loss, _ = run(exp(point), exp(lower), exp(upper), baseline_correct)
        return loss

    left_loss, right_loss = probe(left), probe(right)
    for _ in range(3):
        if left_loss <= right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - ratio * (upper - lower)
            left_loss = probe(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + ratio * (upper - lower)
            right_loss = probe(right)

    reference = trials[0]
    for trial in trials:
        trial["meets_promotion_rule"] = (
            trial["epoch_120_correct"] >= reference["epoch_120_correct"] + 9
            and trial["mean_last_10_val_accuracy"] > reference["mean_last_10_val_accuracy"]
            and trial["mean_last_10_val_loss"] <= baseline_loss
        )
    winner = max(
        (trial for trial in trials if trial["meets_promotion_rule"]),
        key=lambda trial: (trial["epoch_120_correct"], -trial["mean_last_10_val_loss"]),
        default=reference,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trials.csv", trials)
    write_csv(output_dir / "metrics.csv", histories)
    write_csv(output_dir / "recommendation.csv", [winner])
    print(f"Recommended decay {winner['weight_decay']:.12g} at epoch 120")


def save_confusion_matrix(model, inputs, labels, output_dir):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)

    classes = list(range(6))
    counts = confusion_matrix(labels.numpy(), predictions.numpy(), labels=classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true_class": label, **{f"predicted_{index}": count for index, count in enumerate(row)}}
            for label, row in zip(classes, counts)
        ],
    )

    figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay(counts, display_labels=classes).plot(
        ax=axis, cmap="Blues", values_format="d"
    )
    axis.set_title("Validation confusion matrix")
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    print(f"Saved validation confusion matrix in {output_dir.resolve()}")


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


def fit_epochs(experiment, training_data, epochs):
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

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_batches:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        yield epoch, model, loss_fn


def train_experiment(experiment, training_data, val_inputs, val_labels, epochs=120):
    train_inputs, train_labels = training_data.tensors
    history = []
    for epoch, model, loss_fn in fit_epochs(experiment, training_data, epochs):
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


def export_final_model(training_data, json_path):
    from supporting_files.save_model_json import save_model_json

    experiment = selected_large_model(0.01)
    print(f"Final training from fresh weights on {len(training_data)} examples")
    for epoch, model, _ in fit_epochs(experiment, training_data, epochs=120):
        if epoch % 30 == 0:
            print(f"Completed final epoch {epoch}/120", flush=True)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_json(model, hidden_layers=list(experiment.widths), activation="relu", filename=json_path)
    print(f"Saved final model to {json_path.resolve()}")


if __name__ == "__main__":
    main()
