import back_propagation_network as nn
import numpy as np

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)

network = nn.Network(784, 256, 10)

train_size = x_train.shape[0]
batch_size = 100
rate = 0.5


train_accuracy = []
test_accuracy = []
loss = []

for i in range(6000):
    mask = np.random.choice(train_size, 100)
    x_batch = x_train[mask]
    t_batch = t_train[mask]

    grads = network.gradient(x_batch, t_batch)
    for key in ("W1", "W2", "b1", "b2"):
        network.params[key] -= grads[key]

    loss.append(network.loss(x_batch, t_batch))

    # print(f"第{i}次学习 LOSS: {loss[i]:.5f}")
    if i % 50 == 0:
        train_accuracy.append(network.accuracy(x_train, t_train))
        test_accuracy.append(network.accuracy(x_test, t_test))

# for i in test_accuracy:
#     print(f"测试正确率: {i:.4f}")
