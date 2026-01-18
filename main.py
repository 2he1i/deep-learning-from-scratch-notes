import numpy as np
import pickle
import time
from dataset.mnist import load_mnist
from PIL import Image
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"函数 【{func.__name__}】 耗时: {end - start:.6f} 秒")
        return result

    return wrapper


def softmax(x: np.ndarray):
    if x.ndim == 2:
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x / np.sum(x, axis=1, keepdims=True)
    x = np.exp(x - np.max(x))
    return x / np.sum(x)


def sigmoid(x: np.ndarray):
    mask = x < 0
    x[mask] = np.exp(x[mask]) / (1 + np.exp(x[mask]))
    mask = ~mask
    x[mask] = 1 / (1 + np.exp(-x[mask]))

    return x


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False)
img = Image.fromarray(np.uint8(x_train[0].reshape(28, 28)))


def init_network():
    with open("dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


@timer
def forward_propagation(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    z1 = sigmoid(x @ W1 + b1)
    z2 = sigmoid(z1 @ W2 + b2)
    z3 = sigmoid(z2 @ W3 + b3)

    # 性能较低
    # y = np.array([softmax(i) for i in z3])
    # return y

    return softmax(z3)


if __name__ == "__main__":
    print(f"{t_train[0]}")

    network = init_network()
    predict_x = forward_propagation(network, x_test)
    predict_result = np.argmax(predict_x, axis=1)

    # Compute the accuracy of prediction
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(predict_result), batch_size):
        accuracy_cnt += np.sum(
            predict_result[i : i + batch_size] == t_test[i : i + batch_size]
        )

    print(f"Accuracy {(accuracy_cnt / len(t_test)) * 100:.2f} %")
