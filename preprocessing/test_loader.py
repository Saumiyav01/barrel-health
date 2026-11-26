import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew


def load_files_as_windows(ims_dir: Path):
    files = sorted(ims_dir.iterdir())
    windows = []

    for file in files:
        data = np.loadtxt(file)
        windows.append(data)

    windows = np.stack(windows)
    return windows


def create_degradation_labels(num_windows: int):
    return np.linspace(0, 1, num_windows)


def convert_to_stages(labels):
    stages = np.zeros_like(labels, dtype=int)

    stages[(labels >= 0.25) & (labels < 0.50)] = 1
    stages[(labels >= 0.50) & (labels < 0.75)] = 2
    stages[labels >= 0.75] = 3

    return stages


def compute_features(windows):
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    variance = np.var(windows, axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    kurt = kurtosis(windows, axis=1)
    sk = skew(windows, axis=1)

    features = np.concatenate(
        [rms, variance, peak_to_peak, kurt, sk],
        axis=1
    )

    return features


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    ims_dir = project_root / "data" / "raw" / "ims"

    windows = load_files_as_windows(ims_dir)
    labels = create_degradation_labels(windows.shape[0])
    stages = convert_to_stages(labels)

    features = compute_features(windows)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        stages,
        test_size=0.2,
        stratify=stages,
        random_state=42
    )

    # ✅ FEATURE SCALING (VERY IMPORTANT)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Feature shape:", features.shape)
    print(classification_report(y_test, predictions))
