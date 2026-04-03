from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from torch import nn
import torch


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class NNLSBaseline:
    def __init__(self) -> None:
        self.reference_: np.ndarray | None = None
        self.cell_types_: np.ndarray | None = None

    def fit(self, sc_matrix: np.ndarray, sc_labels: np.ndarray) -> "NNLSBaseline":
        cell_types = np.unique(sc_labels)
        profiles = []
        for ct in cell_types:
            profiles.append(sc_matrix[sc_labels == ct].mean(axis=0))
        self.reference_ = np.vstack(profiles)
        self.cell_types_ = cell_types
        return self

    def predict(self, st_matrix: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("NNLSBaseline 尚未拟合。")
        ref = self.reference_.T
        preds = []
        for spot in st_matrix:
            coef, _ = nnls(ref, spot)
            if coef.sum() == 0:
                coef = np.ones_like(coef)
            preds.append(coef / coef.sum())
        return np.vstack(preds).astype(np.float32)
