import numpy as np
from sklearn.datasets import make_classification, make_circles, make_moons, make_blobs, make_gaussian_quantiles


def generate_classification_dataset(dataset_type, n_samples):
    """Generate a classification dataset. Returns X (datapoints), y (labels)"""
    n_classes = 4

    if dataset_type == "classification":
        return make_classification(n_samples=n_samples, n_features=2, n_classes=n_classes,
                                   n_clusters_per_class=1, n_redundant=0)
    elif dataset_type == "aniso":
        return make_classification(n_samples=n_samples, n_features=2, n_classes=n_classes, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, class_sep=2)
    elif dataset_type == "blobs":
        return make_blobs(n_samples=n_samples, centers=n_classes)
    elif dataset_type == "varied_density":
        return make_blobs(n_samples=n_samples, centers=n_classes, cluster_std=np.random.choice([0.5, 1.0, 2.0, 0.1], n_classes))
    elif dataset_type == "outliers_with_clusters":
        X, y = make_blobs(n_samples=n_samples, centers=n_classes)
        X[:20] += 10  # Add outliers
        return X, y
    elif dataset_type == "star_cluster":
        X, y = [], []
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)

        for i, angle in enumerate(angles):
            r = np.random.uniform(0.5, 1.5, n_samples // n_classes)
            x1 = r * np.cos(angle) + np.random.normal(0, 0.2, size=r.shape)
            x2 = r * np.sin(angle) + np.random.normal(0, 0.2, size=r.shape)

            X.append(np.column_stack((x1, x2)))
            y.append(np.full_like(x1, i))

        X = np.vstack(X)
        y = np.concatenate(y).astype(int)
        return X, y
    elif dataset_type == "checkerboard":
        X = np.random.rand(n_samples, 2)
        if n_classes == 4:
            y = []
            for datapoint in X:
                if datapoint[0] < 0.5 and datapoint[1] < 0.5:
                    y.append(0)
                elif datapoint[0] < 0.5 and datapoint[1] >= 0.5:
                    y.append(1)
                elif datapoint[0] >= 0.5 and datapoint[1] < 0.5:
                    y.append(2)
                elif datapoint[0] >= 0.5 and datapoint[1] >= 0.5:
                    y.append(3)
            y = np.asarray(y)
        else:
            y = ((np.floor(X[:, 0] * 2) + np.floor(X[:, 1] * 2)) % n_classes).astype(int)
        return X, y
    elif dataset_type == "concentric_rings":
        radii = np.linspace(0.5, 2.0, n_classes)
        X, y = [], []

        for i, r in enumerate(radii[:n_classes]):  # Support up to 4 classes
            theta = np.linspace(0, 2 * np.pi, n_samples // n_classes)
            x1 = r * np.cos(theta) + np.random.normal(0, 0.1, size=theta.shape)
            x2 = r * np.sin(theta) + np.random.normal(0, 0.1, size=theta.shape)
            X.append(np.column_stack((x1, x2)))
            y.append(np.full_like(x1, i))

        X = np.vstack(X)
        y = np.concatenate(y).astype(int)
        return X, y
    elif dataset_type == "ball":
        return make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=n_classes)
    elif dataset_type == "moons":
        X_1, y_1 = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        X_2, y_2 = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        for idx, datapoint in enumerate(X_2):
            if y_2[idx] == 0:
                datapoint[1] += 1
            else:
                datapoint[1] -= 1
        y_2 += 2

        X = np.concatenate((X_1, X_2), axis=0)
        y = np.concatenate((y_1, y_2), axis=0)
        return X, y
    elif dataset_type == "wavy_clusters":
        X, y = [], []
        x1 = np.linspace(-1, 1, n_samples // n_classes)

        for i in range(n_classes):
            x2 = np.sin(5 * np.pi * x1) + np.random.normal(0, 0.1, size=x1.shape) + 2 * i
            X.append(np.column_stack((x1, x2)))
            y.append(np.full_like(x1, i))

        X = np.vstack(X)
        y = np.concatenate(y).astype(int)
        return X, y
    elif dataset_type == "s_curves":
        x1 = np.linspace(-1, 1, n_samples // n_classes)
        X, y = [], []

        for i in range(n_classes):
            x2 = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, size=x1.shape) + i
            X.append(np.column_stack((x1, x2)))
            y.append(np.full_like(x1, i))

        X = np.vstack(X)
        y = np.concatenate(y).astype(int)
        return X, y
    elif dataset_type == "spiral":
        theta = np.linspace(0, 4 * np.pi, n_samples)
        r = np.linspace(0, 1, n_samples)
        X = np.column_stack([r * np.sin(theta), r * np.cos(theta)])
        y = np.zeros(n_samples, dtype=int)
        if n_classes == 2:
            y[r > 0.5] = 1
        elif n_classes == 3:
            y[r > 0.33] = 1
            y[r > 0.66] = 2
        elif n_classes == 4:
            y[r > 0.25] = 1
            y[r > 0.5] = 2
            y[r > 0.75] = 3
        return X, y
    elif dataset_type == "multiple_spirals":
        n_samples_per_class = n_samples // n_classes
        X, y = [], []

        centers = [(i * 1, i * 1) for i in range(n_classes)]  # Different starting centers

        for i, (cx, cy) in enumerate(centers):
            t = np.linspace(0, 2 * np.pi, n_samples_per_class)  # Spiral shape
            x = cx + t * np.cos(t) + 0.1 * np.random.randn(n_samples_per_class)
            y_coord = cy + t * np.sin(t) + 0.1 * np.random.randn(n_samples_per_class)
            X.append(np.column_stack((x, y_coord)))
            y.append(np.full(n_samples_per_class, i))

        X = np.vstack(X)
        y = np.hstack(y).astype(int)
        return X, y
    elif dataset_type == "multiarm_spiral":
        n_samples_per_class = n_samples // n_classes
        X, y = [], []

        for i in range(n_classes):
            t = np.linspace(0, 3 * 2 * np.pi, n_samples_per_class)  # Spiral shape
            angle_offset = (i / n_classes) * (2 * np.pi)  # Offset each spiral arm
            x = (t + 1) * np.cos(t + angle_offset) + 0.1 * np.random.randn(n_samples_per_class)
            y_coord = (t + 1) * np.sin(t + angle_offset) + 0.1 * np.random.randn(n_samples_per_class)
            X.append(np.column_stack((x, y_coord)))
            y.append(np.full(n_samples_per_class, i))

        X = np.vstack(X)
        y = np.hstack(y).astype(int)
        return X, y
