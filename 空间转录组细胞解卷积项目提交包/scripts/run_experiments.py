from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.config import OUTPUT_ROOT, ensure_output_dirs
from spatial_deconv.data import build_pseudospots, load_dataset_bundle, preprocess_bundle
from spatial_deconv.evaluate import compute_metrics
from spatial_deconv.models import NNLSBaseline
from spatial_deconv.train import train_mlp, train_spatial_gcn
from spatial_deconv.visualize import save_prediction_table, save_spatial_maps, save_training_curve


def _make_graph_adjacency(coords: np.ndarray) -> torch.Tensor:
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=min(7, len(coords))).fit(coords)
    indices = nn_model.kneighbors(return_distance=False)
    adj = np.zeros((len(coords), len(coords)), dtype=np.float32)
    for i, row in enumerate(indices):
        for j in row[1:]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    adj = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    return torch.from_numpy(adj)


def _run_single_experiment(
    dataset: str,
    model_name: str,
    epochs: int,
    num_spots: int,
    max_sc_cells: int,
    device: str,
) -> dict[str, float | str]:
    bundle = load_dataset_bundle(dataset, max_sc_cells=max_sc_cells)
    preprocessed = preprocess_bundle(bundle)
    pseudo = build_pseudospots(preprocessed, num_spots=num_spots)

    if model_name == "mlp":
        artifacts = train_mlp(
            pseudo["train_x"],
            pseudo["train_y"],
            pseudo["val_x"],
            pseudo["val_y"],
            epochs=epochs,
            device=device,
        )
        with torch.no_grad():
            pred = artifacts.model(torch.from_numpy(pseudo["val_x"]).to(device)).cpu().numpy()
    elif model_name == "spatial_gcn":
        artifacts = train_spatial_gcn(
            pseudo["train_x"],
            pseudo["train_y"],
            pseudo["train_coords"],
            pseudo["val_x"],
            pseudo["val_y"],
            pseudo["val_coords"],
            epochs=epochs,
            device=device,
        )
        with torch.no_grad():
            pred, _ = artifacts.model(
                torch.from_numpy(pseudo["val_x"]).to(device),
                _make_graph_adjacency(pseudo["val_coords"]).to(device),
            )
            pred = pred.cpu().numpy()
    elif model_name == "nnls":
        artifacts = None
        pred = NNLSBaseline().fit(preprocessed.sc_matrix, preprocessed.sc_labels).predict(pseudo["val_x"])
    else:
        raise ValueError(f"未知模型: {model_name}")

    metrics = compute_metrics(pseudo["val_y"], pred)
    metrics["dataset"] = dataset
    metrics["model"] = model_name
    metrics["epochs"] = epochs
    metrics["num_spots"] = num_spots
    metrics["max_sc_cells"] = max_sc_cells

    if artifacts is not None:
        model_path = OUTPUT_ROOT / "models" / f"{dataset}_{model_name}.pt"
        torch.save(
            {
                "model_type": model_name,
                "state_dict": artifacts.model.state_dict(),
                "input_dim": int(pseudo["train_x"].shape[1]),
                "output_dim": int(pseudo["train_y"].shape[1]),
                "cell_types": pseudo["cell_types"].tolist(),
                "genes": pseudo["genes"].tolist(),
            },
            model_path,
        )
        figure_path = OUTPUT_ROOT / "figures" / f"{dataset}_{model_name}_loss.png"
        save_training_curve(artifacts.history, figure_path)
        history_path = OUTPUT_ROOT / "logs" / f"{dataset}_{model_name}_history.json"
        history_path.write_text(json.dumps(artifacts.history, indent=2), encoding="utf-8")

        if dataset == "human_lymph_node":
            with torch.no_grad():
                x = torch.from_numpy(preprocessed.st_matrix).to(device)
                if model_name == "mlp":
                    st_pred = artifacts.model(x).cpu().numpy()
                else:
                    st_pred, _ = artifacts.model(x, torch.from_numpy(preprocessed.adjacency).to(device))
                    st_pred = st_pred.cpu().numpy()
            st_pred /= np.maximum(st_pred.sum(axis=1, keepdims=True), 1e-8)
            pred_path = OUTPUT_ROOT / "predictions" / f"{dataset}_{model_name}_predictions.csv"
            save_prediction_table(preprocessed.st_obs_names, st_pred, pseudo["cell_types"].tolist(), pred_path)
            save_spatial_maps(
                preprocessed.st_coords,
                st_pred,
                pseudo["cell_types"].tolist(),
                OUTPUT_ROOT / "figures" / f"{dataset}_{model_name}",
            )

    return metrics


def _write_markdown_summary(df: pd.DataFrame, output_path: Path) -> None:
    ordered = df[["dataset", "model", "mae", "rmse", "pcc", "epochs", "num_spots", "max_sc_cells"]].copy()
    output_path.write_text(ordered.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行正式实验并输出汇总结果。")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["human_lymph_node", "simulated_seqfish"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["nnls", "mlp", "spatial_gcn"],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-spots", type=int, default=1200)
    parser.add_argument("--max-sc-cells", type=int, default=6000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ensure_output_dirs()
    rows: list[dict[str, float | str]] = []
    for dataset in args.datasets:
        for model_name in args.models:
            print(f"running dataset={dataset} model={model_name}")
            rows.append(
                _run_single_experiment(
                    dataset=dataset,
                    model_name=model_name,
                    epochs=args.epochs,
                    num_spots=args.num_spots,
                    max_sc_cells=args.max_sc_cells,
                    device=args.device,
                )
            )

    frame = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    csv_path = OUTPUT_ROOT / "metrics" / "experiment_summary.csv"
    json_path = OUTPUT_ROOT / "metrics" / "experiment_summary.json"
    md_path = OUTPUT_ROOT / "metrics" / "experiment_summary.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_summary(frame, md_path)
    print(f"实验汇总已保存到 {csv_path}")


if __name__ == "__main__":
    main()
