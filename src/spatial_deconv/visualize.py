from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    plt.figure(figsize=(6, 4))
    # Add markers so single-epoch runs are still visible.
    plt.plot(epochs, train_loss, label="train", marker="o")
    plt.plot(epochs, val_loss, label="val", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_spatial_maps(
    coords: np.ndarray,
    proportions: np.ndarray,
    cell_types: list[str],
    output_dir: Path,
    top_n: int = 2,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_props = proportions.mean(axis=0)
    order = np.argsort(mean_props)[::-1][: min(top_n, len(cell_types))]
    paths: list[Path] = []
    for idx in order:
        path = output_dir / f"spatial_{cell_types[idx]}.png"
        plt.figure(figsize=(5, 4))
        plt.scatter(coords[:, 0], coords[:, 1], c=proportions[:, idx], s=12, cmap="viridis")
        plt.colorbar(label=f"{cell_types[idx]} proportion")
        plt.title(cell_types[idx])
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths


def save_prediction_table(
    spot_names: np.ndarray,
    proportions: np.ndarray,
    cell_types: list[str],
    output_path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(proportions, columns=cell_types)
    frame.insert(0, "spot_id", spot_names)
    frame.to_csv(output_path, index=False)
    return frame
