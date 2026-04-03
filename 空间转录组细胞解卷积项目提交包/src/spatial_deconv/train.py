from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spatial_deconv.models import MLPBaseline, SpatialGCN


@dataclass
class TrainArtifacts:
    model: nn.Module
    history: list[dict[str, float]]


def _mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean_diff = source.mean(dim=0) - target.mean(dim=0)
    return torch.mean(mean_diff * mean_diff)


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
) -> TrainArtifacts:
    model = MLPBaseline(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val = torch.from_numpy(val_x).to(device)
    y_val = torch.from_numpy(val_y).to(device)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_batch)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()
        history.append({"epoch": float(epoch), "train_loss": total / len(train_x), "val_loss": val_loss})
    return TrainArtifacts(model=model, history=history)


def train_spatial_gcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_coords: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_coords: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    use_mmd: bool = False,
    device: str = "cpu",
) -> TrainArtifacts:
    model = SpatialGCN(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def make_adjacency(coords: np.ndarray) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=min(7, len(coords))).fit(coords)
        indices = nn_model.kneighbors(return_distance=False)
        adj = np.zeros((len(coords), len(coords)), dtype=np.float32)
        for i, row in enumerate(indices):
            for j in row[1:]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        np.fill_diagonal(adj, 1.0)
        deg = adj.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        return torch.from_numpy(adj / deg).to(device)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    a_train = make_adjacency(train_coords)
    x_val = torch.from_numpy(val_x).to(device)
    y_val = torch.from_numpy(val_y).to(device)
    a_val = make_adjacency(val_coords)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, hidden = model(x_train, a_train)
        loss = criterion(pred, y_train)
        if use_mmd:
            model.eval()
            with torch.no_grad():
                val_hidden = model.encode(x_val, a_val)
            model.train()
            loss = loss + 0.1 * _mmd_loss(hidden, val_hidden)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, _ = model(x_val, a_val)
            val_loss = criterion(pred_val, y_val).item()
        history.append({"epoch": float(epoch), "train_loss": float(loss.item()), "val_loss": val_loss})
    return TrainArtifacts(model=model, history=history)
