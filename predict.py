"""
predict.py
Single-sample inference - given material parameters and frequency,
returns the predicted attenuation in dB/m.
"""

import numpy as np
import joblib
from tensorflow import keras

MODEL_PATH  = "attenuation_model.h5"
SCALER_PATH = "scaler.pkl"


def predict_attenuation(sigma: float, eps_r: float, mu_r: float, freq_hz: float) -> float:
    """
    Predict attenuation alpha (dB/m) for a single set of material parameters.

    Parameters
    ----------
    sigma   : electrical conductivity (S/m)
    eps_r   : relative permittivity
    mu_r    : relative permeability
    freq_hz : frequency in Hz  (e.g. 2.4e9 for 2.4 GHz)

    Returns
    -------
    alpha_db : float - predicted attenuation (dB/m)
    """
    model  = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    inp        = np.array([[np.log10(sigma), eps_r, mu_r, np.log10(freq_hz)]])
    inp_scaled = scaler.transform(inp)
    log_alpha  = model.predict(inp_scaled, verbose=0)[0][0]
    return float(10 ** log_alpha)


if __name__ == "__main__":
    result = predict_attenuation(sigma=0.3, eps_r=25.0, mu_r=1.0, freq_hz=2.4e9)
    print(f"Predicted attenuation: {result:.4f} dB/m")

# Quick CLI usage:
#   python predict.py
# Frequency examples:
#   2.4 GHz Wi-Fi  -> 2.4e9
#   5.0 GHz Wi-Fi  -> 5.0e9
#   1.0 GHz radar  -> 1.0e9
#   100 MHz FM     -> 1.0e8

# Import and call predict_attenuation() directly from other scripts.
