# recommender/evaluation.py
import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    recommended_items: list of item_ids sorted by predicted relevance
    relevant_items: set or list of items the user actually liked (ground truth)
    """
    if not recommended_items:
        return 0.0

    top_k = recommended_items[:k]
    relevant_items = set(relevant_items)

    hits = sum(1 for i in top_k if i in relevant_items)
    return float(hits / min(k, len(recommended_items)))
