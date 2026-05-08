"""
evaluate.py
Loads the saved model and test set, computes R2, MAE, RMSE
and generates a predicted-vs-actual scatter plot.
Output: pred_vs_actual.png
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODEL_PATH  = "attenuation_model.h5"
SCALER_PATH = "scaler.pkl"
PLOT_PATH   = "pred_vs_actual.png"


def evaluate():
    model  = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    y_pred = model.predict(X_test).flatten()

    y_actual    = 10 ** y_test
    y_predicted = 10 ** y_pred

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f} dB/m")
    print(f"RMSE : {rmse:.4f} dB/m")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.3, s=5, color="steelblue", label="Test samples")
    lims = [y_actual.min(), y_actual.max()]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Analytical alpha (dB/m)")
    plt.ylabel("AI Predicted alpha (dB/m)")
    plt.title(f"Predicted vs Actual  |  R2 = {r2:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"Scatter plot saved -> {PLOT_PATH}")
    return r2, mae, rmse


if __name__ == "__main__":
    evaluate()

# R2 computed in log10 space; MAE/RMSE reported in dB/m for interpretability.
