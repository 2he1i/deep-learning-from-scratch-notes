import numpy as np

from collections import OrderedDict

import layers


class Network:
    def __init__(self, input_size, hidden_size, output_size, weigh_init_std=0.01):
        self.params = {}

        self.params["W1"] = weigh_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weigh_init_std * np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()

        # 给对象传入字典时, 实则是引用, 既可以修改原字典, 也可以修改对象的属性使得参数更新
        self.layers["Affine1"] = layers.Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = layers.Relu()
        self.layers["Affine2"] = layers.Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = layers.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():  # 这里是各个层即Layer对象
            x = layer.forward(x)
        y = x

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / x.shape[0]

        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())  #   指向Layer对象
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["W2"] = self.layers["Affine2"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["b2"] = self.layers["Affine2"].db

        return grads
