import numpy as np
import pandas as pd


def ranked_probability_score(y_true_prob: pd.DataFrame, y_pred_prob: pd.DataFrame):
    """
    Args:
        y_true_prob: array of indicators where 1 is placed in the category that
        actually realised, others are 0
        y_pred_prob: array of probability predictions

    Returns: ranked probability score

    """
    y_true_prob = np.cumsum(y_true_prob, axis=1)
    y_pred_prob = np.cumsum(y_pred_prob, axis=1)
    rps = np.sum((y_pred_prob - y_true_prob) ** 2, axis=1) / (y_true_prob.shape[1] - 1)
    return rps.sum() / len(rps)
