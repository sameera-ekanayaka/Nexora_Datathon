"""
src/utils.py
Common evaluation metrics, model persistence, and formatting utilities.
Team: Nexora
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true, y_pred, model_name="Model"):
    """
    Computes standard regression performance metrics:
    RMSE, MAE, R², and Mean Absolute Percentage Error (MAPE).
    
    Robust against pandas Series index mismatches and shape differences.
    """
    # Convert to flat 1D numpy arrays to prevent pandas index alignment errors
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    # Drop any NaN or Inf pairs if present
    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.all(valid_mask):
        dropped = int(np.sum(~valid_mask))
        print(f"[!] Warning: Dropped {dropped} non-finite prediction/target values in {model_name}")
        y_true_arr = y_true_arr[valid_mask]
        y_pred_arr = y_pred_arr[valid_mask]

    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    # Non-zero safe MAPE calculation
    nonzero_mask = y_true_arr > 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_true_arr[nonzero_mask])) * 100.0
    else:
        mape = 0.0

    metrics = {
        "Model": model_name,
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "MAPE (%)": round(float(mape), 2)
    }
    return metrics


def display_metrics_table(metrics_list):
    """Formats a list of metric dicts as a clean pandas DataFrame."""
    df = pd.DataFrame(metrics_list)
    return df


def save_model(model, filepath):
    """Serializes a trained model artifact to .pkl."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Model successfully saved to {filepath} ({size_mb:.2f} MB)")


def load_model(filepath):
    """Loads a serialized model artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
    model = joblib.load(filepath)
    print(f"[+] Model loaded from {filepath}")
    return model
