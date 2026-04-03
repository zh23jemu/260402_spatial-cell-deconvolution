from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_deconv.data import build_pseudospots, load_dataset_bundle, preprocess_bundle


def main() -> None:
    bundle = load_dataset_bundle("simulated_seqfish", max_sc_cells=2000)
    preprocessed = preprocess_bundle(bundle, n_top_genes=500)
    pseudo = build_pseudospots(preprocessed, num_spots=200)
    assert pseudo["train_x"].shape[1] == 500
    assert pseudo["train_y"].shape[1] == len(pseudo["cell_types"])
    sums = pseudo["train_y"].sum(axis=1)
    assert ((sums > 0.999) & (sums < 1.001)).all()
    print("smoke test passed")


if __name__ == "__main__":
    main()
