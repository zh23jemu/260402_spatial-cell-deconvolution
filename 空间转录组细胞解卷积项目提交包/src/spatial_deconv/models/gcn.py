from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(adjacency @ x)


class SpatialGCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gcn1 = GraphConv(hidden_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        hidden = torch.relu(self.gcn1(hidden, adjacency))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.gcn2(hidden, adjacency))
        return hidden

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(x, adjacency)
        pred = torch.softmax(self.head(hidden), dim=-1)
        return pred, hidden
