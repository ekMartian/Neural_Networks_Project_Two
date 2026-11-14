#!/usr/bin/env python3
"""
BITS F445 - Programming Assignment 2
JSON checker and validation-set evaluator.

Usage:
    python check_model_json.py BITSID.json

Required files in the current working directory:
    training.csv
    validation.csv

The program:
    1. validates the JSON structure and parameter shapes;
    2. computes feature means and population standard deviations from training.csv;
    3. standardizes validation.csv using those training-set statistics;
    4. reconstructs the submitted MLP in PyTorch;
    5. loads the submitted weights and biases; and
    6. reports classification accuracy on validation.csv.

Important:
    This program is intended primarily as a format and sanity checker.
    If the final submitted model has been retrained using training.csv +
    validation.csv, the reported validation accuracy is no longer an
    independent estimate of generalization performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


INPUT_SIZE = 24
OUTPUT_SIZE = 6
ALLOWED_HIDDEN_SIZES = {8, 16, 32, 64, 128}
# 'linear' is accepted so that the supplied example JSON can demonstrate
# the file format without suggesting a preferred nonlinear activation.
ALLOWED_ACTIVATIONS = {"linear", "sigmoid", "tanh", "relu"}

FEATURE_COLUMNS = [f"x{i}" for i in range(1, INPUT_SIZE + 1)]
TARGET_COLUMN = "condition"


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,
        activation: str,
    ) -> None:
        super().__init__()

        dims = [input_size, *hidden_layers, output_size]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Hidden-layer activation only; no activation after output layer.
            if i < len(dims) - 2:
                if activation == "linear":
                    layers.append(nn.Identity())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON syntax: {exc}")
    except OSError as exc:
        fail(f"Could not read JSON file: {exc}")

    require(isinstance(data, dict), "Top-level JSON value must be an object.")
    return data


def validate_model_json(
    data: dict,
) -> tuple[list[int], str, list[np.ndarray], list[np.ndarray]]:
    required_keys = {
        "input_size",
        "hidden_layers",
        "activation",
        "output_size",
        "weights",
        "biases",
    }

    missing = required_keys - set(data.keys())
    extra = set(data.keys()) - required_keys

    require(not missing, f"Missing required JSON key(s): {sorted(missing)}")
    require(not extra, f"Unexpected JSON key(s): {sorted(extra)}")

    require(
        isinstance(data["input_size"], int)
        and not isinstance(data["input_size"], bool),
        "'input_size' must be an integer.",
    )
    require(
        data["input_size"] == INPUT_SIZE,
        f"'input_size' must be {INPUT_SIZE}.",
    )

    require(
        isinstance(data["output_size"], int)
        and not isinstance(data["output_size"], bool),
        "'output_size' must be an integer.",
    )
    require(
        data["output_size"] == OUTPUT_SIZE,
        f"'output_size' must be {OUTPUT_SIZE}.",
    )

    hidden = data["hidden_layers"]
    require(isinstance(hidden, list), "'hidden_layers' must be a list.")
    require(
        1 <= len(hidden) <= 3,
        "'hidden_layers' must contain 1 to 3 hidden layers.",
    )

    for i, width in enumerate(hidden, start=1):
        require(
            isinstance(width, int) and not isinstance(width, bool),
            f"Hidden-layer size {i} must be an integer.",
        )
        require(
            width in ALLOWED_HIDDEN_SIZES,
            f"Hidden-layer size {width} is not allowed. "
            f"Allowed sizes: {sorted(ALLOWED_HIDDEN_SIZES)}.",
        )

    activation = data["activation"]
    require(isinstance(activation, str), "'activation' must be a string.")
    require(
        activation in ALLOWED_ACTIVATIONS,
        f"'activation' must be one of {sorted(ALLOWED_ACTIVATIONS)}.",
    )

    weights_raw = data["weights"]
    biases_raw = data["biases"]

    require(isinstance(weights_raw, list), "'weights' must be a list.")
    require(isinstance(biases_raw, list), "'biases' must be a list.")

    dims = [INPUT_SIZE, *hidden, OUTPUT_SIZE]
    n_linear_layers = len(dims) - 1

    require(
        len(weights_raw) == n_linear_layers,
        f"Expected {n_linear_layers} weight matrices, found {len(weights_raw)}.",
    )
    require(
        len(biases_raw) == n_linear_layers,
        f"Expected {n_linear_layers} bias vectors, found {len(biases_raw)}.",
    )

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for layer_idx in range(n_linear_layers):
        in_features = dims[layer_idx]
        out_features = dims[layer_idx + 1]

        try:
            w = np.asarray(weights_raw[layer_idx], dtype=np.float64)
        except (TypeError, ValueError):
            fail(f"Weight matrix {layer_idx + 1} contains non-numeric values.")

        try:
            b = np.asarray(biases_raw[layer_idx], dtype=np.float64)
        except (TypeError, ValueError):
            fail(f"Bias vector {layer_idx + 1} contains non-numeric values.")

        expected_w_shape = (out_features, in_features)
        expected_b_shape = (out_features,)

        require(
            w.shape == expected_w_shape,
            f"Weight matrix {layer_idx + 1} has shape {w.shape}; "
            f"expected {expected_w_shape}.",
        )
        require(
            b.shape == expected_b_shape,
            f"Bias vector {layer_idx + 1} has shape {b.shape}; "
            f"expected {expected_b_shape}.",
        )

        require(
            np.isfinite(w).all(),
            f"Weight matrix {layer_idx + 1} contains NaN or infinity.",
        )
        require(
            np.isfinite(b).all(),
            f"Bias vector {layer_idx + 1} contains NaN or infinity.",
        )

        weights.append(w.astype(np.float32))
        biases.append(b.astype(np.float32))

    return hidden, activation, weights, biases


def read_dataset(path: Path, require_target: bool) -> tuple[np.ndarray, np.ndarray | None]:
    require(path.is_file(), f"Required CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {path.name}: {exc}")

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    require(
        not missing_features,
        f"{path.name} is missing feature column(s): {missing_features}",
    )

    try:
        x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        fail(f"Feature columns in {path.name} must contain numeric values.")

    require(
        x.ndim == 2 and x.shape[1] == INPUT_SIZE,
        f"Feature matrix in {path.name} has an invalid shape.",
    )
    require(len(x) > 0, f"{path.name} contains no examples.")
    require(
        np.isfinite(x).all(),
        f"Features in {path.name} contain NaN or infinity.",
    )

    if not require_target:
        return x, None

    require(
        TARGET_COLUMN in df.columns,
        f"{path.name} is missing the '{TARGET_COLUMN}' column.",
    )

    try:
        y_numeric = pd.to_numeric(
            df[TARGET_COLUMN], errors="raise"
        ).to_numpy(dtype=np.float64)
    except Exception:
        fail(
            f"'{TARGET_COLUMN}' in {path.name} must contain integer "
            "class labels 0,...,5."
        )

    require(
        np.isfinite(y_numeric).all(),
        f"'{TARGET_COLUMN}' in {path.name} contains NaN or infinity.",
    )
    require(
        np.all(np.equal(y_numeric, np.floor(y_numeric))),
        f"'{TARGET_COLUMN}' in {path.name} must contain integer class labels.",
    )

    y = y_numeric.astype(np.int64)
    require(
        np.all((0 <= y) & (y < OUTPUT_SIZE)),
        f"'{TARGET_COLUMN}' labels in {path.name} must be in the range 0,...,5.",
    )

    return x, y


def compute_training_statistics(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)

    require(
        np.isfinite(mean).all() and np.isfinite(std).all(),
        "Training-set statistics contain invalid values.",
    )
    require(
        np.all(std > 1e-12),
        "At least one training feature has zero (or near-zero) standard deviation.",
    )

    return mean, std


def load_parameters(
    model: nn.Module,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> None:
    linear_layers = [
        module for module in model.modules()
        if isinstance(module, nn.Linear)
    ]

    require(
        len(linear_layers) == len(weights),
        "Internal error: reconstructed model does not match JSON.",
    )

    with torch.no_grad():
        for layer, w, b in zip(linear_layers, weights, biases):
            layer.weight.copy_(torch.from_numpy(w))
            layer.bias.copy_(torch.from_numpy(b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a PA2 JSON model and compute its accuracy on "
            "validation.csv using normalization statistics from training.csv."
        )
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the PA2 JSON model file.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    training_path = cwd / "training.csv"
    validation_path = cwd / "validation.csv"

    data = load_json(args.json_file)
    hidden, activation, weights, biases = validate_model_json(data)

    x_train, _ = read_dataset(training_path, require_target=False)
    x_val, y_val = read_dataset(validation_path, require_target=True)
    assert y_val is not None

    mean, std = compute_training_statistics(x_train)
    x_val_std = (x_val - mean) / std

    require(
        np.isfinite(x_val_std).all(),
        "Standardized validation data contain invalid values.",
    )

    model = MLP(
        input_size=INPUT_SIZE,
        hidden_layers=hidden,
        output_size=OUTPUT_SIZE,
        activation=activation,
    )
    load_parameters(model, weights, biases)
    model.eval()

    x_tensor = torch.from_numpy(x_val_std.astype(np.float32))
    y_tensor = torch.from_numpy(y_val)

    with torch.inference_mode():
        logits = model(x_tensor)
        predictions = logits.argmax(dim=1)
        correct = int((predictions == y_tensor).sum().item())
        total = int(y_tensor.numel())
        accuracy = correct / total

    architecture = " -> ".join(
        str(v) for v in [INPUT_SIZE, *hidden, OUTPUT_SIZE]
    )

    print("\nPA2 MODEL JSON CHECK: PASSED")
    print("----------------------------------------")
    print(f"JSON file          : {args.json_file}")
    print(f"Training file      : {training_path}")
    print(f"Validation file    : {validation_path}")
    print(f"Architecture       : {architecture}")
    print(f"Activation         : {activation}")
    print(f"Validation examples: {total}")
    print(f"Correct predictions: {correct}")
    print(
        f"Validation accuracy: "
        f"{accuracy:.6f} ({100.0 * accuracy:.2f}%)"
    )
    print("----------------------------------------")

    if activation == "linear":
        print(
            "Note: 'linear' is accepted so that the supplied example JSON "
            "can demonstrate the file format. Follow the PA2 specification "
            "for the activation functions permitted in the final submission."
        )

    print(
        "Note: If this model was retrained using training.csv + "
        "validation.csv, this validation accuracy is only a sanity check "
        "and is not an independent estimate of generalization performance."
    )


if __name__ == "__main__":
    main()
