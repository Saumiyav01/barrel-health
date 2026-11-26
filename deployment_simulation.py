# deployment_simulation.py

import torch
import numpy as np
from scipy.stats import kurtosis, skew
from models.resnet_bilstm import HybridResNetBiLSTM


MAX_EXPECTED_CYCLES = 6000


# --------------------------------------------------
# Time-Series Normalization
# --------------------------------------------------

def normalize_timeseries(x: torch.Tensor):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-6
    return (x - mean) / std


# --------------------------------------------------
# Stable Statistical Features
# --------------------------------------------------

def safe_stat_features(full_window):

    rms = np.sqrt(np.mean(full_window ** 2, axis=0))
    variance = np.var(full_window, axis=0)
    peak_to_peak = np.ptp(full_window, axis=0)

    kurt = kurtosis(full_window, axis=0, fisher=True, bias=False)
    skw = skew(full_window, axis=0, bias=False)

    # Replace NaNs safely
    kurt = np.nan_to_num(kurt)
    skw = np.nan_to_num(skw)

    return np.concatenate([rms, variance, peak_to_peak, kurt, skw])


# --------------------------------------------------
# Simulate Live Window (Stable Version)
# --------------------------------------------------

def simulate_live_window(previous_cycles):

    progress = previous_cycles / MAX_EXPECTED_CYCLES

    vibration = np.random.normal(0, 0.2, (1024, 8))

    for _ in range(np.random.randint(2, 5)):
        idx = np.random.randint(0, 1015)
        vibration[idx:idx+5] += np.random.uniform(1.5, 3.0) * progress

    # Add small variation to temperature to avoid zero variance
    temp = np.ones((1024, 1)) * (30 + 60 * progress)
    temp += np.random.normal(0, 0.5, temp.shape)

    energy = np.sqrt(np.mean(vibration**2, axis=1)).reshape(-1, 1)

    pseudo_strain = energy * temp
    stress = energy * np.gradient(temp.squeeze()).reshape(-1, 1)

    full_window = np.concatenate(
        [vibration, temp, pseudo_strain, stress],
        axis=1
    )

    stats = safe_stat_features(full_window)

    normalized_cycles = previous_cycles / MAX_EXPECTED_CYCLES
    stats = np.concatenate([stats, [normalized_cycles]])

    return full_window.astype(np.float32), stats.astype(np.float32)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_health_and_rul(previous_cycles):

    window, stats = simulate_live_window(previous_cycles)

    window = torch.tensor(window).permute(1, 0).unsqueeze(0)
    stats = torch.tensor(stats).unsqueeze(0)

    window = normalize_timeseries(window)

    with torch.no_grad():
        health_ratio = model(window, stats).item()

    # Safety guard
    if np.isnan(health_ratio):
        health_ratio = 0.0

    health_ratio = max(0.0, min(1.0, health_ratio))

    remaining_cycles = int(health_ratio * MAX_EXPECTED_CYCLES)
    predicted_total_life = previous_cycles + remaining_cycles

    return health_ratio, remaining_cycles, predicted_total_life


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading trained model...")

model = HybridResNetBiLSTM(stat_dim=56)
model.load_state_dict(torch.load("best_rul_model.pth", map_location="cpu"))
model.eval()

print("Model ready.\n")


# --------------------------------------------------
# Test Scenarios
# --------------------------------------------------

test_cycles = [500, 1500, 3000, 4500, 5500]

for cycles in test_cycles:

    health, remaining, total_life = predict_health_and_rul(cycles)

    print(f"Previous Cycles Used : {cycles}")
    print(f"Wear Health (0-1)    : {health:.4f}")
    print(f"Remaining Cycles     : {remaining}")
    print(f"Predicted Total Life : {total_life}")
    print("-" * 50)

print("\nDeployment simulation completed successfully.")
