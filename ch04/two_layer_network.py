import numpy as np
import functions


class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size).reshape(1, hidden_size)
        self.params["b2"] = np.zeros(output_size).reshape(1, output_size)

    # x: Input data
    def predict(self, x: np.ndarray):
        z1 = functions.sigmoid(x @ self.params["W1"] + self.params["b1"])
        y = functions.softmax(z1 @ self.params["W2"] + self.params["b2"])

        return y

    # x: Iutput data t: Test data
    def loss(self, x, t):
        return np.sum(functions.cross_entropy_error(self.predict(x), t))

    def numerical_gradient(self, x, t, batch):
        def loss(W):
            return self.loss(x, t) / batch

        grads = {}
        grads["W1"] = functions.numerical_gradient(loss, self.params["W1"])
        grads["W2"] = functions.numerical_gradient(loss, self.params["W2"])
        grads["b1"] = functions.numerical_gradient(loss, self.params["b1"])
        grads["b2"] = functions.numerical_gradient(loss, self.params["b2"])

        return grads
