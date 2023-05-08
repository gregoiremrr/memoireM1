from activations import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as func


class Wasserstein_net(nn.Module):
    def __init__(self, n_in, n_h, n_out, activation=func.relu):
        super(Wasserstein_net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_h)
        self.fc4 = nn.Linear(n_h, n_h)
        self.fc5 = nn.Linear(n_h, n_h)
        self.fc6 = nn.Linear(n_h, n_out)
        self.act = activation

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

    def train_(self, nb_epoch, mc_size, learning_rate=1e-2):
        num_epoch = 0
        dist = 1
        while num_epoch <= nb_epoch:
            x1 = torch.normal(mean=0., std=torch.tensor(1.).to(torch.float32), size=(mc_size, 1))
            x2 = torch.normal(mean=0., std=torch.tensor(2.).to(torch.float32), size=(mc_size, 1))

            dist = torch.mean(self.forward(x1)) - torch.mean(self.forward(x2))
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
        print("Ended on epoch {} with dist {}".format(num_epoch, dist.item()))


r = func.relu
d = deepspline.DeepSpline()
g2 = groupsort.GroupSort(num_units=2)
g4 = groupsort.GroupSort(num_units=4)

net = Wasserstein_net(n_in=1, n_h=100, n_out=1, activation=g2)
net.train_(nb_epoch=6000, mc_size=3000, learning_rate=1e-2)
