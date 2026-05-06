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

## Quick Start

```bash
pip install -r requirements.txt
python generate_dataset.py
python preprocess.py
python train_model.py
python evaluate.py
python predict.py
```

## Model Architecture

```
Input (4 features)
  - Dense(64, ReLU)
    - Dense(64, ReLU)
      - Dense(32, ReLU)
        - Dense(1)  <- predicts log10(alpha)
```

Trained with Adam optimizer, MSE loss, 200 epochs, batch size 64.

## Dataset Statistics
- **Total samples**: ~50,000 (5 media x 20 sigma x 20 eps_r x 25 freq)
- **Frequency range**: 1 MHz - 10 GHz
- **alpha range**: ~0.001 - 10,000 dB/m

## Files

| File                  | Purpose                               |
|-----------------------|---------------------------------------|
| `generate_dataset.py` | Analytical alpha sweep -> CSV         |
| `preprocess.py`       | Log-scaling, train/test split, scaler |
| `train_model.py`      | Neural network training               |
| `evaluate.py`         | Metrics + scatter plot                |
| `predict.py`          | Single-sample inference               |
| `run_all.py`          | Full pipeline in one command          |
