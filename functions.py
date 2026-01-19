import numpy as np


def cross_entropy_error(x: np.ndarray, t: np.ndarray):
    assert x.shape == t.shape

    if x.ndim == 1:
        x.reshape(1, x.size)
        t.reshape(1, t.size)

    return -np.sum(t * np.log(x))


def sigmoid(x: np.ndarray):
    mask = x < 0
    x[mask] = np.exp(x[mask]) / (1 + np.exp(x[mask]))
    mask = ~mask
    x[mask] = 1 / (1 + np.exp(-x[mask]))

    return x


if __name__ == "__main__":
    X = np.array([1, 2])
    T = np.array([1, 0])
