import numpy as np
import time
from functools import wraps


# impplement by Gemini
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"函数 【{func.__name__}】 耗时: {end - start:.6f} 秒")
        return result

    return wrapper


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    assert y.shape == t.shape

    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)

    return -np.sum(t * np.log(y + 1e-15))


def sigmoid(x: np.ndarray):
    mask = x < 0
    x[mask] = np.exp(x[mask]) / (1 + np.exp(x[mask]))
    mask = ~mask
    x[mask] = 1 / (1 + np.exp(-x[mask]))

    return x


def softmax(x: np.ndarray):
    if x.ndim == 1:
        x.reshape(1, x.size)
    x_max = x.max(axis=1, keepdims=True)
    return np.exp(x - x_max) / np.sum(np.exp(x - x_max), axis=1, keepdims=True)


# 性能太差
def numerical_gradient(f, x: np.ndarray):
    h = 1e-4
    grads = np.zeros_like(x)
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            bakcup_x = x[j][i]

            x[j][i] += h
            fxh1 = f(x)
            x[j][i] -= 2 * h
            fxh2 = f(x)

            grads[j][i] = (fxh1 - fxh2) / (2 * h)

            x[j][i] = bakcup_x

    return grads


if __name__ == "__main__":
    X = np.array([1, 2])
    T = np.array([1, 0])
