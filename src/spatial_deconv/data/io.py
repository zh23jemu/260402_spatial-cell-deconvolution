from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from spatial_deconv.config import DATASETS


@dataclass
class DatasetBundle:
    name: str
    sc_adata: ad.AnnData
    st_adata: ad.AnnData
    sc_label_column: str
    coords: np.ndarray
    annotation: pd.DataFrame | None = None


def _to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _find_label_column(obs: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in obs.columns and obs[name].nunique(dropna=True) > 1:
            return name

    fallback_names = ["cell_type", "annotation", "labels", "label", "CellType"]
    for name in fallback_names:
        if name in obs.columns and obs[name].nunique(dropna=True) > 1:
            return name

    ranked: list[tuple[int, str]] = []
    for col in obs.columns:
        nunique = obs[col].nunique(dropna=True)
        if 1 < nunique < max(100, len(obs) // 5):
            ranked.append((nunique, col))
    if ranked:
        ranked.sort()
        return ranked[0][1]
    raise ValueError("未找到可用的单细胞标签列。")


def _extract_coords(st_adata: ad.AnnData) -> np.ndarray:
    if "spatial" in st_adata.obsm:
        coords = np.asarray(st_adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]

    obs = st_adata.obs
    for x_name, y_name in [("array_row", "array_col"), ("x", "y"), ("row", "col")]:
        if x_name in obs.columns and y_name in obs.columns:
            return obs[[x_name, y_name]].to_numpy(dtype=float)

    n_spots = st_adata.n_obs
    side = int(np.ceil(np.sqrt(n_spots)))
    rows = np.arange(n_spots) // side
    cols = np.arange(n_spots) % side
    return np.column_stack([rows, cols]).astype(float)


def _attach_spatial_csv(st_adata: ad.AnnData, dataset_root: Path) -> ad.AnnData:
    csv_path = dataset_root / "spatial" / "tissue_positions_list.csv"
    if not csv_path.exists():
        return st_adata

    coords = pd.read_csv(csv_path, header=None)
    if coords.shape[1] >= 6:
        coords.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        coords = coords.set_index("barcode")
        common = st_adata.obs_names.intersection(coords.index)
        if len(common) > 0:
            st_adata = st_adata[common].copy()
            st_adata.obs = st_adata.obs.join(coords.loc[common], how="left")
            st_adata.obsm["spatial"] = st_adata.obs[["pxl_row", "pxl_col"]].to_numpy(dtype=float)
    return st_adata


def _ensure_symbol_var_names(adata: ad.AnnData) -> ad.AnnData:
    if "SYMBOL" in adata.var.columns:
        symbols = adata.var["SYMBOL"].astype(str)
        if symbols.notna().sum() > 0:
            adata = adata[:, symbols != "nan"].copy()
            adata.var_names = pd.Index(symbols[symbols != "nan"], name="gene_symbol")
            adata.var_names_make_unique()
            return adata

    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def _read_10x_h5_as_anndata(path: Path) -> ad.AnnData:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = tuple(matrix["shape"][:])
        barcodes = matrix["barcodes"][:].astype(str)

        features = matrix["features"]
        if "name" in features:
            gene_names = features["name"][:].astype(str)
        elif "gene_names" in features:
            gene_names = features["gene_names"][:].astype(str)
        else:
            gene_names = features["id"][:].astype(str)

        sparse_matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
        adata = ad.AnnData(X=sparse_matrix)
        adata.obs_names = pd.Index(barcodes, name="barcode")
        adata.var_names = pd.Index(gene_names, name="gene")
        adata.var_names_make_unique()
        return adata


def _load_st_adata(path: Path) -> ad.AnnData:
    if path.suffix == ".h5ad":
        return ad.read_h5ad(path)
    if path.suffix == ".h5":
        return _read_10x_h5_as_anndata(path)
    raise ValueError(f"不支持的 ST 文件格式: {path}")


def load_dataset_bundle(dataset_name: str, max_sc_cells: int | None = None) -> DatasetBundle:
    config = DATASETS[dataset_name]
    sc_adata = ad.read_h5ad(config["sc_path"])
    st_adata = _load_st_adata(Path(config["st_path"]))
    st_adata = _attach_spatial_csv(st_adata, Path(config["root"]))

    label_column = _find_label_column(sc_adata.obs, config["label_candidates"])
    sc_adata = _ensure_symbol_var_names(sc_adata)
    st_adata = _ensure_symbol_var_names(st_adata)

    if max_sc_cells is not None and sc_adata.n_obs > max_sc_cells:
        labels = sc_adata.obs[label_column].astype(str)
        sampled: list[str] = []
        for _, group in labels.groupby(labels):
            take = max(1, int(np.ceil(len(group) * max_sc_cells / sc_adata.n_obs)))
            sampled.extend(group.sample(min(take, len(group)), random_state=42).index.tolist())
        sc_adata = sc_adata[sampled].copy()

    annotation = None
    annotation_path = config.get("annotation_path")
    if annotation_path and Path(annotation_path).exists():
        annotation = pd.read_csv(annotation_path)

    coords = _extract_coords(st_adata)
    return DatasetBundle(
        name=dataset_name,
        sc_adata=sc_adata,
        st_adata=st_adata,
        sc_label_column=label_column,
        coords=coords,
        annotation=annotation,
    )


def inspect_dataset(dataset_name: str) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_name)
    sc_labels = bundle.sc_adata.obs[bundle.sc_label_column].astype(str)
    return {
        "dataset": dataset_name,
        "sc_shape": bundle.sc_adata.shape,
        "st_shape": bundle.st_adata.shape,
        "label_column": bundle.sc_label_column,
        "n_cell_types": int(sc_labels.nunique()),
        "top_cell_types": sc_labels.value_counts().head(10).to_dict(),
        "coords_shape": tuple(bundle.coords.shape),
        "annotation_columns": [] if bundle.annotation is None else bundle.annotation.columns.tolist(),
    }


def sc_matrix_and_labels(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    matrix = _to_dense(bundle.sc_adata.X).astype(np.float32)
    labels = bundle.sc_adata.obs[bundle.sc_label_column].astype(str).to_numpy()
    return matrix, labels


def st_matrix(bundle: DatasetBundle) -> np.ndarray:
    return _to_dense(bundle.st_adata.X).astype(np.float32)
