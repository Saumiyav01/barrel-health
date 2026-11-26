# build_dataset.py

import numpy as np
from preprocessing.deep_dataset import IMSDataset
from scipy.stats import kurtosis, skew


# --------------------------------------------------
# Statistical Feature Extraction
# --------------------------------------------------

def compute_statistical_features(windows: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    variance = np.var(windows, axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    kurt = kurtosis(windows, axis=1)
    skw = skew(windows, axis=1)

    return np.concatenate(
        [rms, variance, peak_to_peak, kurt, skw],
        axis=1
    )


# --------------------------------------------------
# Physics Injection
# --------------------------------------------------

def inject_ballistic_simulation(windows: np.ndarray) -> np.ndarray:

    total_samples = len(windows)

    for i in range(total_samples):

        progress = i / total_samples
        window = windows[i]

        # Mild amplitude scaling
        window *= np.random.uniform(0.9, 1.1)

        # Impulses
        num_shots = np.random.randint(1, 4)
        if np.random.rand() < 0.25:
            num_shots = np.random.randint(4, 8)

        for _ in range(num_shots):
            shot_time = np.random.randint(0, 1015)
            impulse_amp = np.random.uniform(1.5, 3.5) * (1 + 0.5 * progress)
            window[shot_time:shot_time+5] += impulse_amp

        # Temperature modeling
        temp = np.zeros(1024)
        baseline = 30 + 40 * progress
        cooling_rate = 0.02 - 0.008 * progress

        for _ in range(num_shots):
            shot_time = np.random.randint(0, 1024)
            decay = 100 * np.exp(-cooling_rate * np.arange(1024 - shot_time))
            temp[shot_time:] += decay

        temp += baseline

        # Wear noise
        noise = np.random.normal(0, 0.03 + 0.2 * progress, window.shape)
        window += noise

        # Physics channels
        vibration_energy = np.sqrt(np.mean(window ** 2, axis=1)).reshape(-1, 1)
        temp = temp.reshape(-1, 1)

        pseudo_strain = vibration_energy * temp
        stress_index = vibration_energy * np.gradient(temp, axis=0)

        windows[i] = np.concatenate(
            [window[:, :8], temp, pseudo_strain, stress_index],
            axis=1
        )

    return windows


# --------------------------------------------------
# RUL + Previous Cycles
# --------------------------------------------------

def compute_rul_and_cycles(total_samples):

    previous_cycles = np.arange(total_samples)
    max_cycles = total_samples

    normalized_cycles = previous_cycles / max_cycles
    rul = (max_cycles - previous_cycles) / max_cycles

    noise = np.random.normal(0, 0.02, rul.shape)
    rul += noise
    rul = np.clip(rul, 0, 1)

    return normalized_cycles.astype(np.float32), rul.astype(np.float32)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Building dataset...")

    dataset = IMSDataset("data/raw/ims")
    windows = dataset.windows

    windows = inject_ballistic_simulation(windows)
    stats = compute_statistical_features(windows)

    normalized_cycles, rul = compute_rul_and_cycles(len(windows))

    normalized_cycles = normalized_cycles.reshape(-1, 1)
    stats = np.concatenate([stats, normalized_cycles], axis=1)

    np.save("dataset_windows.npy", windows)
    np.save("dataset_stats.npy", stats)
    np.save("dataset_rul.npy", rul)

    print("Windows shape:", windows.shape)
    print("Stats shape:", stats.shape)
    print("RUL shape:", rul.shape)
    print("Dataset built successfully.")


if __name__ == "__main__":
    main()
