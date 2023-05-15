from activations import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as func
import pandas as pd


class Wasserstein_net_for_numbers(nn.Module):
    def __init__(self, n_in, n_h, n_out, activation=func.relu):
        super(Wasserstein_net_for_numbers, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_h)
        self.fc4 = nn.Linear(n_h, n_h)
        self.fc5 = nn.Linear(n_h, n_h)
        self.fc6 = nn.Linear(n_h, n_out)
        self.act = activation
        self.dist = -1

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)
        return x

    def normaliseSpectral(self):
        n1 = torch.linalg.vector_norm(self.fc1.weight.data)
        n2 = torch.linalg.matrix_norm(self.fc2.weight.data, ord=2)
        n3 = torch.linalg.matrix_norm(self.fc3.weight.data, ord=2)
        n4 = torch.linalg.matrix_norm(self.fc4.weight.data, ord=2)
        n5 = torch.linalg.matrix_norm(self.fc5.weight.data, ord=2)
        n6 = torch.linalg.vector_norm(self.fc6.weight.data)
        self.fc1.weight.data = self.fc1.weight.data / max(n1, 1)
        self.fc2.weight.data = self.fc2.weight.data / max(n2, 1)
        self.fc3.weight.data = self.fc3.weight.data / max(n3, 1)
        self.fc4.weight.data = self.fc4.weight.data / max(n4, 1)
        self.fc5.weight.data = self.fc5.weight.data / max(n5, 1)
        self.fc6.weight.data = self.fc6.weight.data / max(n6, 1)

    def normaliseSpectralForce(self):
        n1 = torch.linalg.vector_norm(self.fc1.weight.data)
        n2 = torch.linalg.matrix_norm(self.fc2.weight.data, ord=2)
        n3 = torch.linalg.matrix_norm(self.fc3.weight.data, ord=2)
        n4 = torch.linalg.matrix_norm(self.fc4.weight.data, ord=2)
        n5 = torch.linalg.matrix_norm(self.fc5.weight.data, ord=2)
        n6 = torch.linalg.vector_norm(self.fc6.weight.data)
        self.fc1.weight.data = self.fc1.weight.data / n1
        self.fc2.weight.data = self.fc2.weight.data / n2
        self.fc3.weight.data = self.fc3.weight.data / n3
        self.fc4.weight.data = self.fc4.weight.data / n4
        self.fc5.weight.data = self.fc5.weight.data / n5
        self.fc6.weight.data = self.fc6.weight.data / n6

    def train_(self, x, y, nb_epoch, batch_size, learning_rate=1e-2):
        num_epoch = 0
        dist = 1
        rng = np.random.default_rng()
        while num_epoch <= nb_epoch:
            batch_x = rng.integers(low=0, high=len(x), size=batch_size)
            batch_y = rng.integers(low=0, high=len(y), size=batch_size)
            x_batch = x[batch_x]
            y_batch = y[batch_y]

            dist = torch.mean(self.forward(x_batch)) - torch.mean(self.forward(y_batch))
            # Zero gradients, perform a backward pass, and update the weights.
            self.zero_grad()  # re-init the gradients (otherwise they are cumulated)
            dist.backward()  # perform back-propagation
            with torch.no_grad():  # update the weights
                for param in self.parameters():
                    param += learning_rate * param.grad
                self.normaliseSpectral()

            if num_epoch % 500 == 0:
                print("epoch {}, dist {}".format(num_epoch, dist.item()))
            num_epoch += 1
        self.dist = dist.item()
        print("Ended on epoch {} with dist {}".format(num_epoch, dist.item()))


r = func.relu
d = deepspline.DeepSpline()
g2 = groupsort.GroupSort(num_units=2)
g4 = groupsort.GroupSort(num_units=4)

mnist = pd.read_csv("mnist_test.csv")

mnist_sans_label = mnist.drop("label", axis=1)

mnist0 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 0]
mnist0bis = torch.tensor(np.array([e.values for e in mnist0])).to(torch.float32)

# to compute the W distance between the two data sets of 0
mnist0bis_a = mnist0bis[0:400]
mnist0bis_b = mnist0bis[401:]

mnist1 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 1]
mnist1bis = torch.tensor(np.array([e.values for e in mnist1])).to(torch.float32)

mnist1bis_a = mnist1bis[0:400]
mnist1bis_b = mnist1bis[401:]

mnist2 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 2]
mnist2bis = torch.tensor(np.array([e.values for e in mnist2])).to(torch.float32)

mnist3 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 3]
mnist3bis = torch.tensor(np.array([e.values for e in mnist3])).to(torch.float32)

mnist4 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 4]
mnist4bis = torch.tensor(np.array([e.values for e in mnist4])).to(torch.float32)

mnist5 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 5]
mnist5bis = torch.tensor(np.array([e.values for e in mnist5])).to(torch.float32)

mnist6 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 6]
mnist6bis = torch.tensor(np.array([e.values for e in mnist6])).to(torch.float32)

mnist7 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 7]
mnist7bis = torch.tensor(np.array([e.values for e in mnist7])).to(torch.float32)

mnist7bis_a = mnist7bis[0:400]
mnist7bis_b = mnist7bis[401:]

mnist8 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 8]
mnist8bis = torch.tensor(np.array([e.values for e in mnist8])).to(torch.float32)

mnist9 = [mnist_sans_label.iloc[i] for i in range(len(mnist)) if mnist.iloc[i]["label"] == 9]
mnist9bis = torch.tensor(np.array([e.values for e in mnist9])).to(torch.float32)

global_n_in = 784
global_n_h = 100
global_n_out = 1
global_activation = g2

global_nb_epoch = 3000
global_batch_size = 200
global_learning_rate = 1e-2

net = Wasserstein_net_for_numbers(n_in=global_n_in, n_h=global_n_h, n_out=global_n_out, activation=global_activation)
#net.train_(mnist7bis_a, mnist7bis_b, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)

# 0 and ...
net.train_(mnist0bis, mnist1bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d01 = net.dist
net.train_(mnist0bis, mnist2bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d02 = net.dist
net.train_(mnist0bis, mnist3bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d03 = net.dist
net.train_(mnist0bis, mnist4bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d04 = net.dist
net.train_(mnist0bis, mnist5bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d05 = net.dist
net.train_(mnist0bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d06 = net.dist
net.train_(mnist0bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d07 = net.dist
net.train_(mnist0bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d08 = net.dist
net.train_(mnist0bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d09 = net.dist

print("Ended 0")

# 1 and ...
net.train_(mnist1bis, mnist2bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d12 = net.dist
net.train_(mnist1bis, mnist3bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d13 = net.dist
net.train_(mnist1bis, mnist4bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d14 = net.dist
net.train_(mnist1bis, mnist5bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d15 = net.dist
net.train_(mnist1bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d16 = net.dist
net.train_(mnist1bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d17 = net.dist
net.train_(mnist1bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d18 = net.dist
net.train_(mnist1bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d19 = net.dist

print("Ended 1")

# 2 and ...
net.train_(mnist2bis, mnist3bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d23 = net.dist
net.train_(mnist2bis, mnist4bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d24 = net.dist
net.train_(mnist2bis, mnist5bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d25 = net.dist
net.train_(mnist2bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d26 = net.dist
net.train_(mnist2bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d27 = net.dist
net.train_(mnist2bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d28 = net.dist
net.train_(mnist2bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d29 = net.dist

print("Ended 2")

# 3 and ...
net.train_(mnist3bis, mnist4bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d34 = net.dist
net.train_(mnist3bis, mnist5bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d35 = net.dist
net.train_(mnist3bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d36 = net.dist
net.train_(mnist3bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d37 = net.dist
net.train_(mnist3bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d38 = net.dist
net.train_(mnist3bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d39 = net.dist

print("Ended 3")

# 4 and ...
net.train_(mnist4bis, mnist5bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d45 = net.dist
net.train_(mnist4bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d46 = net.dist
net.train_(mnist4bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d47 = net.dist
net.train_(mnist4bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d48 = net.dist
net.train_(mnist4bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d49 = net.dist

print("Ended 4")

# 5 and ...
net.train_(mnist5bis, mnist6bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d56 = net.dist
net.train_(mnist5bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d57 = net.dist
net.train_(mnist5bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d58 = net.dist
net.train_(mnist5bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d59 = net.dist

print("Ended 5")

# 6 and ...
net.train_(mnist6bis, mnist7bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d67 = net.dist
net.train_(mnist6bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d68 = net.dist
net.train_(mnist6bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d69 = net.dist

print("Ended 6")

# 7 and ...
net.train_(mnist7bis, mnist8bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d78 = net.dist
net.train_(mnist7bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d79 = net.dist

print("Ended 7")

# 8 and 9
net.train_(mnist8bis, mnist9bis, nb_epoch=global_nb_epoch, batch_size=global_batch_size, learning_rate=global_learning_rate)
d89 = net.dist

print("Ended 8")

# Results
a0 = [-1, d01, d02, d03, d04, d05, d06, d07, d08, d09]
a1 = [d01, -1, d12, d13, d14, d15, d16, d17, d18, d19]
a2 = [d02, d12, -1, d23, d24, d25, d26, d27, d28, d29]
a3 = [d03, d13, d23, -1, d34, d35, d36, d37, d38, d39]
a4 = [d04, d14, d24, d34, -1, d45, d46, d47, d48, d49]
a5 = [d05, d15, d25, d35, d45, -1, d56, d57, d58, d59]
a6 = [d06, d16, d26, d36, d46, d56, -1, d67, d68, d69]
a7 = [d07, d17, d27, d37, d47, d57, d67, -1, d78, d79]
a8 = [d08, d18, d28, d38, d48, d58, d68, d78, -1, d89]
a9 = [d09, d19, d29, d39, d49, d59, d69, d79, d89, -1]

data = {'0': a0, '1': a1, '2': a2, '3': a3, '4': a4, '5': a5, '6': a6, '7': a7, '8': a8, '9': a9}
df = pd.DataFrame(data, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                  index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

print(df)
