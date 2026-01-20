import numpy as np

import two_layer_network as nw
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)


network = nw.Network(784, 50, 10)
theta = 0.1
batch = 100

for i in range(100):
    print(f"第{i + 1}次学习")
    M = np.random.choice(x_train.shape[0], batch)
    A = x_train[M]

    Dw1 = network.numerical_gradient(A, t_train[M], batch)["W1"]
    Dw2 = network.numerical_gradient(A, t_train[M], batch)["W2"]
    Db1 = network.numerical_gradient(A, t_train[M], batch)["b1"]
    Db2 = network.numerical_gradient(A, t_train[M], batch)["b2"]

    network.params["W1"] -= theta * Dw1
    network.params["W2"] -= theta * Dw2
    network.params["b1"] -= theta * Db1
    network.params["b2"] -= theta * Db2

    print(network.loss(A, t_train[M]) / batch)

acc_cnt = 0

y = np.sum(
    np.array(network.predict(x_test).argmax(axis=1) - t_test.argmax(axis=1) == 0)
)
# 这里应为 x_test.shape[0] 会导致结果 / 784 最终训练结果为 78.4% ~ 86.24%
print(f"accuracy: {(y / x_test.size) * 100:.2f} %")
