import numpy as np


def spiral_data(n_samples: int, n_classes: int) -> tuple[list, list]:
    """
    Generates a spiral dataset.

    Params:
        n_samples (int): Number of sample points.
        n_classes (int): Number of labels.
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype="uint8")
    for class_number in range(n_classes):
        ix = range(n_samples * class_number, n_samples * (class_number + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


def vertical_data(n_samples: int, n_classes: int) -> tuple[list, list]:
    """
    Generates a vertical dataset.

    Params:
        n_samples (int): Number of sample points.
        n_classes (int): Number of labels.
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype="uint8")
    for class_number in range(n_classes):
        ix = range(n_samples * class_number, n_samples * (class_number + 1))
        X[ix] = np.c_[np.random.randn(n_samples) * .1 + (class_number) / 3, np.random.randn(n_samples) * .1 + 0.5]
        y[ix] = class_number
    return X, y


def sine_data(n_samples: int) -> tuple[list, list]:
    """
    Generates a sine dataset.

    Params:
        n_samples (int): Number of sample points.
    """

    X = np.arange(n_samples).reshape(-1, 1) / n_samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)

    return X, y
