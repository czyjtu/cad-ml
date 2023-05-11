import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_score, recall_score, f1_score


# TODO add another metrics? auc?
def calculate_metrics(preds: npt.NDArray[int], golds: npt.NDArray[int]) -> dict[str, float]:
    metrics = {
        'precision': precision_score(golds, preds, zero_division=0).astype(np.float32),
        'recall': recall_score(golds, preds, zero_division=0).astype(np.float32),
        'f1': f1_score(golds, preds, zero_division=0).astype(np.float32),
        'support': np.sum(golds).astype(np.float32)
    }

    return metrics
