"""Simple LSTM-based time series forecasting example.

This module trains a neural network to predict the next value of a
synthetic sine wave time series.  It is self-contained and can be run
from the command line::

    python timeseries_model.py
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 0) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


class SineWaveDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Sliding window dataset over a sine wave."""

    def __init__(self, sequence_length: int, num_samples: int, noise_std: float = 0.0):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.noise_std = noise_std
        self._series = self._generate_series()

    def _generate_series(self) -> torch.Tensor:
        xs = torch.linspace(0, 10 * math.pi, steps=self.num_samples + self.sequence_length)
        ys = torch.sin(xs)
        if self.noise_std > 0:
            ys += torch.randn_like(ys) * self.noise_std
        return ys

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        start = idx
        end = idx + self.sequence_length
        window = self._series[start:end]
        target = self._series[end]
        return window.unsqueeze(-1), target.unsqueeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        lstm_out, _ = self.lstm(inputs)
        last_hidden = lstm_out[:, -1, :]
        return self.output_layer(last_hidden)


@dataclass
class TrainingConfig:
    sequence_length: int = 30
    num_samples: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    noise_std: float = 0.05


def train(config: TrainingConfig) -> LSTMForecaster:
    set_seed()
    dataset = SineWaveDataset(config.sequence_length, config.num_samples, config.noise_std)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        average_loss = epoch_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {average_loss:.4f}")

    return model


def evaluate(model: LSTMForecaster, config: TrainingConfig) -> float:
    dataset = SineWaveDataset(config.sequence_length, config.num_samples, noise_std=0.0)
    loader = DataLoader(dataset, batch_size=config.batch_size)
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            losses.append(criterion(outputs, targets).item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    config = TrainingConfig()
    model = train(config)
    mse = evaluate(model, config)
    print(f"Final evaluation MSE: {mse:.4f}")
