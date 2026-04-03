from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    per_class_pcc = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
            continue
        per_class_pcc.append(pearsonr(y_true[:, i], y_pred[:, i]).statistic)
    pcc = float(np.mean(per_class_pcc)) if per_class_pcc else 0.0
    return {"mae": mae, "rmse": rmse, "pcc": pcc}


def metrics_to_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
