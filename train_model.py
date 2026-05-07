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

# Architecture notes:
# Input (4) -> Dense(64, relu) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)
# Three hidden layers chosen empirically; deeper caused no improvement.
# Output is log10(alpha_dbm) - easier for the network than raw dB/m values.

# Note: batch_size=64 balances GPU utilisation and gradient noise.
