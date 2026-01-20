from abc import ABC, abstractmethod
from functions import softmax, cross_entropy_error
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class Relu(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.m = None

    def forward(self, x: np.ndarray):
        self.m = x < 0
        out = x.copy()
        out[self.m] = 0

        return out

    def backward(self, dout: np.ndarray):
        dout[self.m] = 0
        dx = dout

        return dx


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        super().__init__()
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.x = x
        out = x @ self.W + self.b

        return out

    def backward(self, dout: np.ndarray):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout  # type: ignore
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]  # type: ignore
        dx = (self.y - self.t) / batch_size  # type: ignore

        return dx
