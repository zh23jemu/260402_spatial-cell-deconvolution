from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.config import OUTPUT_ROOT, ensure_output_dirs
from spatial_deconv.data import build_pseudospots, load_dataset_bundle, preprocess_bundle
from spatial_deconv.evaluate import compute_metrics, metrics_to_frame
from spatial_deconv.models import NNLSBaseline
from spatial_deconv.train import train_mlp, train_spatial_gcn
from spatial_deconv.visualize import save_training_curve


def _load_or_build_pseudo(dataset: str, max_sc_cells: int, num_spots: int):
    pseudo_path = OUTPUT_ROOT / "data" / f"{dataset}_pseudospots.npz"
    bundle = load_dataset_bundle(dataset, max_sc_cells=max_sc_cells)
    preprocessed = preprocess_bundle(bundle)
    if pseudo_path.exists():
        pseudo = dict(np.load(pseudo_path, allow_pickle=True))
    else:
        pseudo = build_pseudospots(preprocessed, num_spots=num_spots)
        np.savez_compressed(pseudo_path, **pseudo)
    return pseudo, preprocessed


def main() -> None:
    parser = argparse.ArgumentParser(description="训练解卷积模型。")
    parser.add_argument("--dataset", default="human_lymph_node")
    parser.add_argument("--model", choices=["mlp", "spatial_gcn"], default="spatial_gcn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-spots", type=int, default=4000)
    parser.add_argument("--max-sc-cells", type=int, default=15000)
    parser.add_argument("--use-mmd", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ensure_output_dirs()
    pseudo, preprocessed = _load_or_build_pseudo(args.dataset, args.max_sc_cells, args.num_spots)

    if args.model == "mlp":
        artifacts = train_mlp(
            pseudo["train_x"],
            pseudo["train_y"],
            pseudo["val_x"],
            pseudo["val_y"],
            epochs=args.epochs,
            device=args.device,
        )
    else:
        artifacts = train_spatial_gcn(
            pseudo["train_x"],
            pseudo["train_y"],
            pseudo["train_coords"],
            pseudo["val_x"],
            pseudo["val_y"],
            pseudo["val_coords"],
            epochs=args.epochs,
            use_mmd=args.use_mmd,
            device=args.device,
        )

    model_path = OUTPUT_ROOT / "models" / f"{args.dataset}_{args.model}.pt"
    torch.save(
        {
            "model_type": args.model,
            "state_dict": artifacts.model.state_dict(),
            "input_dim": int(pseudo["train_x"].shape[1]),
            "output_dim": int(pseudo["train_y"].shape[1]),
            "cell_types": pseudo["cell_types"].tolist(),
            "genes": pseudo["genes"].tolist(),
        },
        model_path,
    )

    train_curve_path = OUTPUT_ROOT / "figures" / f"{args.dataset}_{args.model}_loss.png"
    save_training_curve(artifacts.history, train_curve_path)

    with torch.no_grad():
        if args.model == "mlp":
            pred = artifacts.model(torch.from_numpy(pseudo["val_x"]).to(args.device)).cpu().numpy()
        else:
            from sklearn.neighbors import NearestNeighbors

            coords = pseudo["val_coords"]
            nn_model = NearestNeighbors(n_neighbors=min(7, len(coords))).fit(coords)
            indices = nn_model.kneighbors(return_distance=False)
            adj = np.zeros((len(coords), len(coords)), dtype=np.float32)
            for i, row in enumerate(indices):
                for j in row[1:]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            np.fill_diagonal(adj, 1.0)
            adj = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
            pred, _ = artifacts.model(
                torch.from_numpy(pseudo["val_x"]).to(args.device),
                torch.from_numpy(adj).to(args.device),
            )
            pred = pred.cpu().numpy()

    rows = [{"model": args.model, **compute_metrics(pseudo["val_y"], pred)}]
    nnls_pred = NNLSBaseline().fit(preprocessed.sc_matrix, preprocessed.sc_labels).predict(pseudo["val_x"])
    rows.append({"model": "nnls", **compute_metrics(pseudo["val_y"], nnls_pred)})
    metrics_path = OUTPUT_ROOT / "metrics" / f"{args.dataset}_{args.model}_validation.csv"
    metrics_to_frame(rows).to_csv(metrics_path, index=False)

    history_path = OUTPUT_ROOT / "logs" / f"{args.dataset}_{args.model}_history.json"
    history_path.write_text(json.dumps(artifacts.history, indent=2), encoding="utf-8")
    print(f"模型已保存到 {model_path}")
    print(f"验证指标已保存到 {metrics_path}")


if __name__ == "__main__":
    main()
