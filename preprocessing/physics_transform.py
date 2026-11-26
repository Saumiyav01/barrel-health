import numpy as np


def apply_physics_transform(windows, training=True):
    """
    Physics-informed transform.
    During training: adds progressive degradation.
    During inference: neutral transform (no artificial scaling).
    """

    np.random.seed(42)

    n_windows, seq_len, n_channels = windows.shape
    transformed = []

    for i in range(n_windows):

        signal = windows[i].copy()

        if training and n_windows > 1:
            degradation_ratio = i / (n_windows - 1)
        else:
            degradation_ratio = 0.0  # no artificial degradation during inference

        # Progressive vibration amplification (training only)
        amplitude_scale = 1 + 1.5 * degradation_ratio
        signal *= amplitude_scale

        # Noise
        noise_level = 0.02 + 0.08 * degradation_ratio
        noise = np.random.normal(0, noise_level, signal.shape)
        signal += noise

        # Temperature model
        temp = np.zeros(seq_len)

        spike_height = 150 + 350 * degradation_ratio
        baseline = 40 + 120 * degradation_ratio
        decay_rate = 0.005 + 0.01 * degradation_ratio

        temp[0] = baseline + spike_height

        for t in range(1, seq_len):
            temp[t] = temp[t - 1] * np.exp(-decay_rate)

        temp = temp.reshape(-1, 1)

        vib_energy = np.mean(signal ** 2, axis=1, keepdims=True)
        pseudo_strain = vib_energy * temp

        rms = np.sqrt(np.mean(signal ** 2, axis=1, keepdims=True))
        temp_gradient = np.gradient(temp, axis=0)
        stress_index = rms * temp_gradient

        augmented = np.concatenate(
            [signal, temp, pseudo_strain, stress_index],
            axis=1
        )

        transformed.append(augmented)

    return np.stack(transformed)
