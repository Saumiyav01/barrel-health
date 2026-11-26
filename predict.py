import torch
import numpy as np
from pathlib import Path
from models.resnet_bilstm import HybridResNetBiLSTM
from preprocessing.physics_transform import apply_physics_transform
from scipy.stats import kurtosis, skew


def compute_stat_features(windows):
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    variance = np.var(windows, axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    kurt = kurtosis(windows, axis=1)
    skw = skew(windows, axis=1)

    return np.concatenate([rms, variance, peak_to_peak, kurt, skw], axis=1)


def load_and_prepare(file_path):

    data = np.loadtxt(file_path)

    # reshape to (1, seq_len, channels)
    data = data.reshape(1, data.shape[0], data.shape[1])

    # downsample to 1024
    factor = data.shape[1] // 1024
    data = data[:, ::factor, :]

    # apply physics transform
    data = apply_physics_transform(data, training=False)


    # compute statistical features
    stats = compute_stat_features(data)

    # convert to tensors
    windows = torch.tensor(data).permute(0, 2, 1).float()
    stats = torch.tensor(stats).float()

    return windows, stats


def predict(file_path):

    device = torch.device("cpu")

    model = HybridResNetBiLSTM()
    model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
    model.eval()

    windows, stats = load_and_prepare(file_path)

    with torch.no_grad():
        output = model(windows, stats)
        prediction = torch.argmax(output, dim=1).item()

    stage_names = [
        "Healthy",
        "Early Degradation",
        "Moderate Wear",
        "Severe Wear Risk"
    ]

    print("\nPredicted Stage:", stage_names[prediction])


if __name__ == "__main__":
    # Replace with your new test file
    test_file = "data/raw/ims/2003.11.25.23.39.56"
    predict(test_file)
