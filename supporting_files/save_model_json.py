#!/usr/bin/env python3
"""
BITS F445 - Programming Assignment 2
Helper function for saving a trained PyTorch MLP in the required JSON format.

Typical use:

    from save_model_json import save_model_json

    save_model_json(
        model,
        hidden_layers=[64, 32],
        activation="relu",
        filename="2014A0PS1234G.json",
    )

The function assumes:
    - 24 input features;
    - 1 to 3 hidden layers;
    - hidden-layer widths from {8, 16, 32, 64, 128};
    - the same hidden activation in all hidden layers;
    - 6 output logits;
    - nn.Linear layers with biases.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


INPUT_SIZE = 24
OUTPUT_SIZE = 6
ALLOWED_HIDDEN_SIZES = {8, 16, 32, 64, 128}
ALLOWED_ACTIVATIONS = {"sigmoid", "tanh", "relu"}


def save_model_json(
    model: nn.Module,
    hidden_layers: list[int],
    activation: str,
    filename: str | Path,
) -> None:
    """Save a trained PA2 MLP in the required JSON format."""

    if not isinstance(hidden_layers, list):
        raise ValueError("hidden_layers must be a list.")

    if not 1 <= len(hidden_layers) <= 3:
        raise ValueError("Use 1 to 3 hidden layers.")

    for width in hidden_layers:
        if width not in ALLOWED_HIDDEN_SIZES:
            raise ValueError(
                f"Hidden-layer size {width} is not allowed. "
                f"Allowed sizes: {sorted(ALLOWED_HIDDEN_SIZES)}"
            )

    activation = activation.lower()
    if activation not in ALLOWED_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {sorted(ALLOWED_ACTIVATIONS)}"
        )

    linear_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]

    expected_dims = [INPUT_SIZE, *hidden_layers, OUTPUT_SIZE]

    if len(linear_layers) != len(expected_dims) - 1:
        raise ValueError(
            "The number of nn.Linear layers in the model does not match "
            "the supplied hidden_layers list."
        )

    for i, layer in enumerate(linear_layers):
        expected_in = expected_dims[i]
        expected_out = expected_dims[i + 1]

        if layer.in_features != expected_in or layer.out_features != expected_out:
            raise ValueError(
                f"Linear layer {i + 1} has shape "
                f"{layer.in_features} -> {layer.out_features}, but "
                f"{expected_in} -> {expected_out} was expected."
            )

        if layer.bias is None:
            raise ValueError(
                f"Linear layer {i + 1} has no bias. "
                "PA2 models are expected to use biases."
            )

    weights = []
    biases = []

    for i, layer in enumerate(linear_layers, start=1):
        weight = layer.weight.detach().cpu()
        bias = layer.bias.detach().cpu()

        if not torch.isfinite(weight).all():
            raise ValueError(
                f"Weight matrix {i} contains NaN or infinity."
            )

        if not torch.isfinite(bias).all():
            raise ValueError(
                f"Bias vector {i} contains NaN or infinity."
            )

        weights.append(weight.tolist())
        biases.append(bias.tolist())

    data = {
        "input_size": INPUT_SIZE,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "output_size": OUTPUT_SIZE,
        "weights": weights,
        "biases": biases,
    }

    output_path = Path(filename)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    print(
        "This file provides the save_model_json() function.\n"
        "Import it into your PA2 training program; see the example "
        "in the module docstring."
    )
