from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.config import OUTPUT_ROOT, ensure_output_dirs
from spatial_deconv.data import build_pseudospots, load_dataset_bundle, preprocess_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 pseudo-spot 训练数据。")
    parser.add_argument("--dataset", default="human_lymph_node")
    parser.add_argument("--num-spots", type=int, default=4000)
    parser.add_argument("--max-sc-cells", type=int, default=15000)
    args = parser.parse_args()

    ensure_output_dirs()
    bundle = load_dataset_bundle(args.dataset, max_sc_cells=args.max_sc_cells)
    preprocessed = preprocess_bundle(bundle)
    pseudo = build_pseudospots(preprocessed, num_spots=args.num_spots)
    output_path = OUTPUT_ROOT / "data" / f"{args.dataset}_pseudospots.npz"
    np.savez_compressed(output_path, **pseudo)
    print(f"已保存 pseudo-spot 数据到 {output_path}")


if __name__ == "__main__":
    main()
