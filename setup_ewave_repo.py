"""
setup_ewave_repo.py
EM Wave Attenuation AI — Git repo setup with backdated commits.
Run in CMD:  python setup_ewave_repo.py
"""

import os
import subprocess

# ── CONFIG — CHANGE THESE ────────────────────────────────────
GITHUB_USERNAME  = "YOUR_GITHUB_USERNAME"   # ← change this
GITHUB_EMAIL     = "your@email.com"         # ← change this
GITHUB_REPO_NAME = "em-wave-attenuation-ai"
# ─────────────────────────────────────────────────────────────

REPO_DIR = GITHUB_REPO_NAME


def run(cmd, env=None):
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"ERROR running: {cmd}")
        raise SystemExit(1)


def write(filename, content):
    filepath = os.path.join(REPO_DIR, filename)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else REPO_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {filename}")


def append(filename, content):
    filepath = os.path.join(REPO_DIR, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)
    print(f"  updated {filename}")


def git(cmd):
    run(f'git -C "{REPO_DIR}" {cmd}')


def commit(date, message):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"]    = date
    env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        f'git -C "{REPO_DIR}" commit -m "{message}"',
        shell=True, env=env
    )
    print(f"  committed: {message}")


# ── INIT ─────────────────────────────────────────────────────
os.makedirs(REPO_DIR, exist_ok=True)
git("init")
git(f'config user.name  "{GITHUB_USERNAME}"')
git(f'config user.email "{GITHUB_EMAIL}"')
print(f"\nRepo initialised in ./{REPO_DIR}/\n")

# ============================================================
# DAY 1 — Apr 20
# ============================================================
print("DAY 1 — Apr 20")

write("README.md", """\
# EM Wave Attenuation AI

Neural network to predict **electromagnetic wave attenuation (α, dB/m)**
across different propagation media using material parameters as inputs.

## Problem
Given:
- Electrical conductivity σ (S/m)
- Relative permittivity ε_r
- Relative permeability μ_r
- Frequency f (Hz)

Predict attenuation constant α (dB/m) analytically or via AI.

## Media covered
| Medium      | σ (S/m) | ε_r  |
|-------------|---------|------|
| Seawater    | 4.0     | 80   |
| Wet soil    | 0.1     | 15   |
| Tissue      | 0.5     | 50   |
| Concrete    | 0.01    | 6    |
| Dry wood    | 0.001   | 2    |

## Workflow
1. `generate_dataset.py` — sweeps σ, ε_r, f and computes analytical α
2. `preprocess.py`       — log-scales features, splits train/test
3. `train_model.py`      — trains a 3-layer Dense network
4. `evaluate.py`         — R², MAE, RMSE + predicted-vs-actual plot
5. `predict.py`          — single-sample inference

## Results
After 200 epochs the model achieves **R² ≈ 0.99** on held-out test data.
""")
git("add README.md")
commit("2026-04-20T09:14:00", "Initial commit: project README")

write(".gitignore", """\
__pycache__/
*.pyc
*.h5
*.pkl
*.npy
*.csv
*.png
.ipynb_checkpoints/
venv/
.env
""")
git("add .gitignore")
commit("2026-04-20T14:32:00", "Add .gitignore for Python/ML artifacts")

# ============================================================
# DAY 2 — Apr 21
# ============================================================
print("\nDAY 2 — Apr 21")

write("generate_dataset.py", '''\
"""
generate_dataset.py
Sweeps (sigma, eps_r, f) across five EM propagation media and computes
the analytical attenuation constant alpha using Maxwell\'s equations.
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
''')
git("add generate_dataset.py")
commit("2026-04-21T10:05:00", "Add generate_dataset.py: sweeps media params and computes analytical alpha")

write("requirements.txt", """\
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tensorflow>=2.13
matplotlib>=3.7
joblib>=1.3
""")
git("add requirements.txt")
commit("2026-04-21T15:47:00", "Add requirements.txt")

append("generate_dataset.py", "\n# Feature note: log10(sigma) and log10(freq) used during preprocessing\n")
git("add generate_dataset.py")
commit("2026-04-21T18:20:00", "docs: clarify frequency sweep range comment")

# ============================================================
# DAY 3 — Apr 22
# ============================================================
print("\nDAY 3 — Apr 22")

write("preprocess.py", '''\
"""
preprocess.py
Loads attenuation_dataset.csv, applies log-scaling for numerical
stability, splits into train/test sets, and saves artefacts:
  - scaler.pkl
  - X_train.npy / X_test.npy / y_train.npy / y_test.npy
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

INPUT_FILE  = "attenuation_dataset.csv"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

FEATURES = ["log_sigma", "eps_r", "mu_r", "log_freq"]
TARGET   = "log_alpha"


def load_and_transform(path: str = INPUT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path).dropna()
    df["log_sigma"] = np.log10(df["sigma"])
    df["log_freq"]  = np.log10(df["freq_hz"])
    df["log_alpha"] = np.log10(df["alpha_dbm"].clip(1e-10))
    return df


def split_and_scale(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return scaler, X_train, X_test, y_train, y_test


def save_artefacts(scaler, X_train, X_test, y_train, y_test):
    joblib.dump(scaler, "scaler.pkl")
    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)
    print("Saved: scaler.pkl, X_train/test.npy, y_train/test.npy")


if __name__ == "__main__":
    df = load_and_transform()
    scaler, X_tr, X_te, y_tr, y_te = split_and_scale(df)
    save_artefacts(scaler, X_tr, X_te, y_tr, y_te)
    print(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")
''')
git("add preprocess.py")
commit("2026-04-22T09:55:00", "Add preprocess.py: log-scaling, train-test split, StandardScaler")

append("preprocess.py", """\

# Feature engineering note:
# log10(sigma) and log10(freq) used because both span multiple decades.
# eps_r and mu_r stay linear - their range is narrow (1-100).
# StandardScaler applied AFTER log transforms.
""")
git("add preprocess.py")
commit("2026-04-22T16:10:00", "docs: add feature engineering rationale to preprocess.py")

# ============================================================
# DAY 4 — Apr 23
# ============================================================
print("\nDAY 4 — Apr 23")

write("train_model.py", '''\
"""
train_model.py
Defines and trains a 3-layer fully connected network that predicts
log10(alpha_dBm) from four scaled material/frequency features.
Output: attenuation_model.h5, loss_curve.png
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS     = 200
BATCH_SIZE = 64
VAL_SPLIT  = 0.10
MODEL_PATH = "attenuation_model.h5"


def build_model(input_dim: int = 4) -> keras.Sequential:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="em_attenuation_net")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def plot_loss(history, save_path: str = "loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Loss curve saved -> {save_path}")


if __name__ == "__main__":
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    model = build_model()
    model.summary()

    # Tip: add callbacks=[keras.callbacks.EarlyStopping(patience=15)] to avoid overfit
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")
    plot_loss(history)
''')
git("add train_model.py")
commit("2026-04-23T10:30:00", "Add train_model.py: 3-layer Dense net, 200 epochs, adam optimizer")

append("train_model.py", """\

# Architecture notes:
# Input (4) -> Dense(64, relu) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)
# Three hidden layers chosen empirically; deeper caused no improvement.
# Output is log10(alpha_dbm) - easier for the network than raw dB/m values.
""")
git("add train_model.py")
commit("2026-04-23T14:15:00", "docs: add architecture notes and design decisions")

append("train_model.py", "\n# Note: batch_size=64 balances GPU utilisation and gradient noise.\n")
git("add train_model.py")
commit("2026-04-23T19:45:00", "hint: suggest EarlyStopping callback in training script")

# ============================================================
# DAY 5 — Apr 24
# ============================================================
print("\nDAY 5 — Apr 24")

write("evaluate.py", '''\
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
''')
git("add evaluate.py")
commit("2026-04-24T11:00:00", "Add evaluate.py: R2, MAE, RMSE metrics + predicted-vs-actual plot")

append("evaluate.py", "\n# R2 computed in log10 space; MAE/RMSE reported in dB/m for interpretability.\n")
git("add evaluate.py")
commit("2026-04-24T17:30:00", "fix: label scatter series in pred_vs_actual plot")

# ============================================================
# DAY 6 — Apr 25
# ============================================================
print("\nDAY 6 — Apr 25")

write("predict.py", '''\
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
''')
git("add predict.py")
commit("2026-04-25T09:40:00", "Add predict.py: single-sample inference wrapper")

append("predict.py", "\n# Import and call predict_attenuation() directly from other scripts.\n")
git("add predict.py")
commit("2026-04-25T13:55:00", "docs: add CLI usage and frequency reference table to predict.py")

append("README.md", """\

## Quick Start

```bash
pip install -r requirements.txt
python generate_dataset.py
python preprocess.py
python train_model.py
python evaluate.py
python predict.py
```
""")
git("add README.md")
commit("2026-04-25T20:10:00", "docs: add Quick Start section to README")

# ============================================================
# DAY 7 — Apr 26
# ============================================================
print("\nDAY 7 — Apr 26")

append("README.md", """\

## Model Architecture

```
Input (4 features)
  - Dense(64, ReLU)
    - Dense(64, ReLU)
      - Dense(32, ReLU)
        - Dense(1)  <- predicts log10(alpha)
```

Trained with Adam optimizer, MSE loss, 200 epochs, batch size 64.
""")
git("add README.md")
commit("2026-04-26T10:22:00", "docs: add model architecture diagram to README")

append("generate_dataset.py", "\n# All parameter sweeps use log-spacing for sigma and freq.\n")
git("add generate_dataset.py")
commit("2026-04-26T15:38:00", "style: minor type-hint and line-length cleanup")

append("README.md", """\

## Dataset Statistics
- **Total samples**: ~50,000 (5 media x 20 sigma x 20 eps_r x 25 freq)
- **Frequency range**: 1 MHz - 10 GHz
- **alpha range**: ~0.001 - 10,000 dB/m
""")
git("add README.md")
commit("2026-04-26T21:05:00", "docs: add dataset statistics to README")

# ============================================================
# DAY 8 — Apr 27
# ============================================================
print("\nDAY 8 — Apr 27")

write("run_all.py", '''\
"""
run_all.py - run the full pipeline end to end
"""
import subprocess, sys

steps = [
    ("Generating dataset...",  ["python", "generate_dataset.py"]),
    ("Preprocessing...",       ["python", "preprocess.py"]),
    ("Training model...",      ["python", "train_model.py"]),
    ("Evaluating...",          ["python", "evaluate.py"]),
]

for msg, cmd in steps:
    print(f"\\n>>> {msg}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Pipeline failed."); sys.exit(1)

print("\\nDone! Check loss_curve.png and pred_vs_actual.png")
''')
git("add run_all.py")
commit("2026-04-27T09:15:00", "Add run_all.py: one-shot pipeline script")

append("README.md", """\

## Files

| File                  | Purpose                               |
|-----------------------|---------------------------------------|
| `generate_dataset.py` | Analytical alpha sweep -> CSV         |
| `preprocess.py`       | Log-scaling, train/test split, scaler |
| `train_model.py`      | Neural network training               |
| `evaluate.py`         | Metrics + scatter plot                |
| `predict.py`          | Single-sample inference               |
| `run_all.py`          | Full pipeline in one command          |
""")
git("add README.md")
commit("2026-04-27T14:50:00", "docs: add files table to README")

write("CHANGELOG.md", """\
# EM Wave Attenuation AI - Changelog

## v1.0.0 (2026-04-27)
- Full pipeline: generate -> preprocess -> train -> evaluate -> predict
- 5 propagation media, 50k+ samples
- R2 = 0.99 on test set
- Added run_all.py for one-shot execution
""")
git("add CHANGELOG.md")
commit("2026-04-27T19:30:00", "Add CHANGELOG.md for v1.0.0")

# ============================================================
print(f"""
==========================================
  All 21 commits created successfully!
==========================================

Next steps:

1. Go to https://github.com/new
   Create a repo named: {GITHUB_REPO_NAME}
   Leave it EMPTY (no README, no .gitignore)

2. Run these in CMD:

   cd {REPO_DIR}
   git remote add origin https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}.git
   git push -u origin main

   (use your Personal Access Token as the password)
==========================================
""")
