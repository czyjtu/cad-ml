import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_fscore_support


# TODO add another metrics? auc?
def calculate_metrics(preds: np.NDArray[int], golds: npt.NDArray[int]) -> dict[str, float]:
    pr, rc, f1, support = \
        precision_recall_fscore_support(golds, preds, average='binary', zero_division=0)

    metrics = {
        'precision': pr,
        'recall': rc,
        'f1': f1,
        'support': support
    }

    return metrics
