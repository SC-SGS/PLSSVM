import numpy as np


def generate_regression_dataset(dataset_type, n_samples):
    """Generate a regression dataset. Returns X (datapoints), y (labels)"""
    if dataset_type == "linear":
        X = np.sort(np.random.rand(n_samples, 1), axis=0)
        y = np.copy(X).ravel()
    elif dataset_type == "quadratic":
        X = np.sort(np.random.rand(n_samples, 1), axis=0)
        y = (X ** 2).ravel()
    elif dataset_type == "cubic":
        X = np.sort(np.random.rand(n_samples, 1), axis=0)
        y = (X ** 3).ravel()
    elif dataset_type == "v":
        X = np.sort(np.random.rand(n_samples, 1), axis=0)
        y = (np.abs(X - 0.5) + 0.5).ravel()
    elif dataset_type == "step":
        X = np.sort(np.random.rand(n_samples, 1), axis=0)
        y = (X > 0.5).astype(int).ravel()
    elif dataset_type == "sine":
        X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
        y = np.sin(X).ravel()
    elif dataset_type == "tanh":
        X = np.sort(10 * np.random.rand(n_samples, 1) - 5, axis=0)
        y = np.tanh(X).ravel()
    elif dataset_type == "1/x":
        X_1 = np.sort(np.random.uniform(-2.5, -0.05, n_samples // 2).reshape((n_samples // 2, 1)), axis=0)
        X_2 = np.sort(np.random.uniform(0.05, 2.5, n_samples // 2).reshape((n_samples // 2, 1)), axis=0)
        X = np.concatenate((X_1, X_2), axis=0)
        y = np.asarray([1 / x for x in X]).ravel()
    elif dataset_type == "irregular":
        X = np.sort(np.random.uniform(-3, 3, n_samples).reshape((n_samples, 1)), axis=0)
        y = np.asarray([
            0.1 * x ** 5 - 0.5 * x ** 3  # Polynomial base
            + 2 * np.exp(-0.2 * (x - 2) ** 2)  # Gaussian bump on the right
            - 3 * np.exp(-0.5 * (x + 1) ** 2)  # Sharp dip on the left
            + 1 / (x ** 2 + 1)  # Rational term to create non-linearity
            for x in X
        ]).ravel()
    else:
        # unreachable
        X = []
        y = []
    return X, y


def generate_noisy_regression_dataset(dataset_type, n_samples, noise):
    """Generate a noisy regression dataset. Returns X (datapoints), y (labels)"""
    X, y = generate_regression_dataset(dataset_type, n_samples)
    num_noisy_datapoints = int(n_samples * noise)
    if num_noisy_datapoints > 0:
        noise_level = n_samples // num_noisy_datapoints
        y[::noise_level] += np.max(X) * (0.5 - np.random.rand(num_noisy_datapoints))  # add noise to every 5th datapoint
    return X, y
