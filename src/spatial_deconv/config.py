from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "Data"
OUTPUT_ROOT = ROOT / "outputs"

DATASETS = {
    "human_lymph_node": {
        "root": DATA_ROOT / "4.Human_Lymph_Node",
        "sc_path": DATA_ROOT / "4.Human_Lymph_Node" / "scRNA.h5ad",
        "st_path": DATA_ROOT / "4.Human_Lymph_Node" / "ST.h5ad",
        "label_candidates": ["cell_type", "Subset", "CellType", "Subset_Broad", "PrelimCellType"],
        "annotation_path": DATA_ROOT / "4.Human_Lymph_Node" / "manual_GC_annot.csv",
    },
    "simulated_seqfish": {
        "root": DATA_ROOT / "11.Simulated_seqFISH+",
        "sc_path": DATA_ROOT / "11.Simulated_seqFISH+" / "scRNA.h5ad",
        "st_path": DATA_ROOT / "11.Simulated_seqFISH+" / "Spatial.h5ad",
        "label_candidates": ["celltype_final", "cell_type", "annotation"],
        "annotation_path": None,
    },
    "human_breast_cancer": {
        "root": DATA_ROOT / "3.Human_Breast_Cancer",
        "sc_path": DATA_ROOT / "3.Human_Breast_Cancer" / "scRNA.h5ad",
        "st_path": DATA_ROOT / "3.Human_Breast_Cancer" / "filtered_feature_bc_matrix.h5",
        "label_candidates": ["cell_type", "CellType", "annotation", "labels"],
        "annotation_path": None,
    },
}


def ensure_output_dirs() -> None:
    for subdir in [
        OUTPUT_ROOT / "data",
        OUTPUT_ROOT / "models",
        OUTPUT_ROOT / "metrics",
        OUTPUT_ROOT / "predictions",
        OUTPUT_ROOT / "figures",
        OUTPUT_ROOT / "logs",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
