"""
generate_dataset.py
Sweeps (sigma, eps_r, f) across five EM propagation media and computes
the analytical attenuation constant alpha using Maxwell's equations.
Output: attenuation_dataset.csv
"""

import numpy as np
import pandas as pd
import itertools


def compute_alpha(sigma: float, eps_r: float, mu_r: float, freq_hz: float) -> float:
    """
    Compute attenuation constant alpha (Np/m) from material parameters.

    Parameters
    ----------
    sigma   : electrical conductivity (S/m)
    eps_r   : relative permittivity
    mu_r    : relative permeability
    freq_hz : frequency (Hz)

    Returns
    -------
    alpha : float - attenuation in Np/m  (multiply by 8.686 for dB/m)
    """
    eps0  = 8.854e-12
    mu0   = 4 * np.pi * 1e-7
    omega = 2 * np.pi * freq_hz
    eps   = eps_r * eps0
    mu    = mu_r  * mu0

    loss_tangent = sigma / (omega * eps)
    term  = np.sqrt(1 + loss_tangent**2) - 1
    alpha = omega * np.sqrt(mu * eps / 2) * np.sqrt(term)
    return alpha


MEDIA = {
    "seawater": (4.0,  80.0, 1.0),
    "soil_wet": (0.1,  15.0, 1.0),
    "tissue":   (0.5,  50.0, 1.0),
    "concrete": (0.01,  6.0, 1.0),
    "dry_wood": (0.001, 2.0, 1.0),
}

OUTPUT_FILE = "attenuation_dataset.csv"


def generate(n_sigma: int = 20, n_eps: int = 20, n_freq: int = 25) -> pd.DataFrame:
    rows = []
    for name, (sigma_base, eps_r_base, mu_r_base) in MEDIA.items():
        sigmas = np.logspace(np.log10(sigma_base * 0.5),
                             np.log10(sigma_base * 2.0), n_sigma)
        eps_rs = np.linspace(eps_r_base * 0.8, eps_r_base * 1.2, n_eps)
        freqs  = np.logspace(6, 10, n_freq)   # 1 MHz - 10 GHz (log-spaced)

        for sigma, eps_r, freq in itertools.product(sigmas, eps_rs, freqs):
            alpha = compute_alpha(sigma, eps_r, mu_r_base, freq)
            rows.append({
                "medium":    name,
                "sigma":     sigma,
                "eps_r":     eps_r,
                "mu_r":      mu_r_base,
                "freq_hz":   freq,
                "alpha_npm": alpha,
                "alpha_dbm": alpha * 8.686,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df):,} samples -> {OUTPUT_FILE}")

# Feature note: log10(sigma) and log10(freq) used during preprocessing

# All parameter sweeps use log-spacing for sigma and freq.
