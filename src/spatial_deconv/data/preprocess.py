from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from spatial_deconv.data.io import DatasetBundle, sc_matrix_and_labels, st_matrix


@dataclass
class PreprocessedBundle:
    dataset_name: str
    sc_matrix: np.ndarray
    sc_labels: np.ndarray
    st_matrix: np.ndarray
    st_coords: np.ndarray
    genes: np.ndarray
    cell_types: list[str]
    adjacency: np.ndarray
    st_obs_names: np.ndarray


def _normalize_log1p(matrix: np.ndarray) -> np.ndarray:
    lib = matrix.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    matrix = matrix / lib * 1e4
    return np.log1p(matrix)


def _select_hvg(sc_matrix: np.ndarray, st_matrix: np.ndarray, genes: np.ndarray, n_top: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.vstack([sc_matrix, st_matrix])
    variances = combined.var(axis=0)
    n_take = min(n_top, len(genes))
    idx = np.argsort(variances)[::-1][:n_take]
    return sc_matrix[:, idx], st_matrix[:, idx], genes[idx]


def _knn_adjacency(coords: np.ndarray, k: int = 6) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(return_distance=False)
    adjacency = np.zeros((len(coords), len(coords)), dtype=np.float32)
    for i, row in enumerate(indices):
        for j in row[1:]:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1.0
    return adjacency / degree


def preprocess_bundle(bundle: DatasetBundle, n_top_genes: int = 2000, knn_k: int = 6) -> PreprocessedBundle:
    sc = bundle.sc_adata.copy()
    st = bundle.st_adata.copy()

    shared = np.intersect1d(sc.var_names.astype(str), st.var_names.astype(str))
    if len(shared) == 0:
        raise ValueError("scRNA 与 ST 没有共享基因。")
    if len(shared) < n_top_genes:
        n_top_genes = min(1000, len(shared))

    sc = sc[:, shared].copy()
    st = st[:, shared].copy()
    inner_bundle = DatasetBundle(bundle.name, sc, st, bundle.sc_label_column, bundle.coords, bundle.annotation)
    sc_matrix, sc_labels = sc_matrix_and_labels(inner_bundle)
    st_expr = st_matrix(inner_bundle)

    sc_matrix = _normalize_log1p(sc_matrix)
    st_expr = _normalize_log1p(st_expr)
    genes = shared.astype(str)
    sc_matrix, st_expr, genes = _select_hvg(sc_matrix, st_expr, genes, n_top=n_top_genes)

    cell_types = sorted(pd.Series(sc_labels).astype(str).unique().tolist())
    adjacency = _knn_adjacency(bundle.coords, k=knn_k)
    return PreprocessedBundle(
        dataset_name=bundle.name,
        sc_matrix=sc_matrix.astype(np.float32),
        sc_labels=sc_labels.astype(str),
        st_matrix=st_expr.astype(np.float32),
        st_coords=bundle.coords.astype(np.float32),
        genes=genes,
        cell_types=cell_types,
        adjacency=adjacency.astype(np.float32),
        st_obs_names=st.obs_names.to_numpy(),
    )


def build_pseudospots(
    preprocessed: PreprocessedBundle,
    num_spots: int = 4000,
    min_cells: int = 3,
    max_cells: int = 8,
    train_size: float = 0.8,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    label_to_indices: dict[str, np.ndarray] = {}
    for cell_type in preprocessed.cell_types:
        idx = np.where(preprocessed.sc_labels == cell_type)[0]
        if len(idx) > 0:
            label_to_indices[cell_type] = idx

    cell_types = list(label_to_indices.keys())
    n_genes = preprocessed.sc_matrix.shape[1]
    pseudo_x = np.zeros((num_spots, n_genes), dtype=np.float32)
    pseudo_y = np.zeros((num_spots, len(cell_types)), dtype=np.float32)
    pseudo_coords = rng.uniform(0.0, 1.0, size=(num_spots, 2)).astype(np.float32)

    for i in range(num_spots):
        n_pick = int(rng.integers(min_cells, max_cells + 1))
        n_types = int(rng.integers(1, min(4, len(cell_types)) + 1))
        chosen_types = rng.choice(cell_types, size=n_types, replace=False)
        weights = rng.dirichlet(np.ones(len(chosen_types)))
        counts = np.maximum(1, np.round(weights * n_pick).astype(int))
        counts[-1] += n_pick - counts.sum()

        expr_parts = []
        for ct, count in zip(chosen_types, counts):
            sampled = rng.choice(label_to_indices[ct], size=max(1, count), replace=True)
            expr_parts.append(preprocessed.sc_matrix[sampled].mean(axis=0))
            pseudo_y[i, cell_types.index(ct)] = max(1, count)
        pseudo_x[i] = np.mean(expr_parts, axis=0)

    pseudo_y /= pseudo_y.sum(axis=1, keepdims=True)
    train_idx, val_idx = train_test_split(np.arange(num_spots), train_size=train_size, random_state=random_state, shuffle=True)
    return {
        "train_x": pseudo_x[train_idx],
        "train_y": pseudo_y[train_idx],
        "train_coords": pseudo_coords[train_idx],
        "val_x": pseudo_x[val_idx],
        "val_y": pseudo_y[val_idx],
        "val_coords": pseudo_coords[val_idx],
        "cell_types": np.asarray(cell_types),
        "genes": preprocessed.genes,
    }
