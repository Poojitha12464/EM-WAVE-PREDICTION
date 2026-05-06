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

# Feature engineering note:
# log10(sigma) and log10(freq) used because both span multiple decades.
# eps_r and mu_r stay linear - their range is narrow (1-100).
# StandardScaler applied AFTER log transforms.
