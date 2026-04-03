from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.config import OUTPUT_ROOT, ensure_output_dirs
from spatial_deconv.data import load_dataset_bundle, preprocess_bundle
from spatial_deconv.models import MLPBaseline, SpatialGCN
from spatial_deconv.visualize import save_prediction_table, save_spatial_maps


def _load_model(checkpoint_path: Path, device: str):
    payload = torch.load(checkpoint_path, map_location=device)
    if payload["model_type"] == "mlp":
        model = MLPBaseline(payload["input_dim"], payload["output_dim"])
    else:
        model = SpatialGCN(payload["input_dim"], payload["output_dim"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="在真实 ST 上做推理并输出空间图。")
    parser.add_argument("--dataset", default="human_lymph_node")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-sc-cells", type=int, default=15000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ensure_output_dirs()
    bundle = load_dataset_bundle(args.dataset, max_sc_cells=args.max_sc_cells)
    preprocessed = preprocess_bundle(bundle)
    model, payload = _load_model(Path(args.checkpoint), args.device)
    x = torch.from_numpy(preprocessed.st_matrix).to(args.device)

    with torch.no_grad():
        if payload["model_type"] == "mlp":
            pred = model(x).cpu().numpy()
        else:
            adj = torch.from_numpy(preprocessed.adjacency).to(args.device)
            pred, _ = model(x, adj)
            pred = pred.cpu().numpy()

    pred /= np.maximum(pred.sum(axis=1, keepdims=True), 1e-8)
    pred_path = OUTPUT_ROOT / "predictions" / f"{args.dataset}_{payload['model_type']}_predictions.csv"
    save_prediction_table(preprocessed.st_obs_names, pred, payload["cell_types"], pred_path)
    figure_dir = OUTPUT_ROOT / "figures" / f"{args.dataset}_{payload['model_type']}"
    save_spatial_maps(preprocessed.st_coords, pred, payload["cell_types"], figure_dir)
    print(f"预测结果已保存到 {pred_path}")
    print(f"空间图已保存到 {figure_dir}")


if __name__ == "__main__":
    main()
