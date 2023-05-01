import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as func

from network import scalar_network


class WGAN(nn.Module):
    def __init__(self, activation, n_in, n_h):
        super(WGAN, self).__init__()
        self.generator = scalar_network.Net(10, n_h, n_in, activation=func.relu)
        self.discriminator = scalar_network.Net(n_in, n_h, 1, activation=activation)

    def train_(self, x, nb_epoch, batch_size, n_critic, learning_rate=1e-2, normalise='s'):

        num_epoch = 0
        rng = np.random.default_rng()

        while num_epoch <= nb_epoch:

            for i in range(n_critic):
                batch = rng.integers(low=0, high=len(x), size=batch_size)

                z = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=batch_size)
                a1 = self.discriminator(x[batch])
                a2 = self.generator(z)
                a3 = self.discriminator(a2.detach().numpy())
                discriminator_criterion = torch.mean(a1 - a3)
                self.zero_grad()
                discriminator_criterion.backward()
                with torch.no_grad():  # update the weights
                    for param in self.discriminator.parameters():
                        param += learning_rate * param.grad
                    if normalise == 's':
                        self.discriminator.normaliseSpectral()
                    elif normalise == 'i':
                        self.discriminator.normaliseInfty()

            z = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=batch_size)
            generator_criterion = torch.mean(self.discriminator(self.generator(z).detach().numpy()))
            self.zero_grad()
            generator_criterion.backward()
            with torch.no_grad():  # update the weights
                for param in self.discriminator.parameters():
                    param += learning_rate * param.grad

            if num_epoch % 10 == 0:
                print("epoch {}, discriminator_criterion {}, generator_criterion {}".format(num_epoch, discriminator_criterion.item(), generator_criterion.item()))

            num_epoch += 1

    def train2_(self, x, nb_epoch, batch_size, n_critic, learning_rate=1e-2, normalise='s'):

        num_epoch = 0
        rng = np.random.default_rng()

        while num_epoch <= nb_epoch:

            for i in range(n_critic):

                z = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=batch_size)
                generator_criterion = torch.mean(self.discriminator(self.generator(z).detach().numpy()))
                self.zero_grad()
                generator_criterion.backward()
                with torch.no_grad():  # update the weights
                    for param in self.discriminator.parameters():
                        param += learning_rate * param.grad

            batch = rng.integers(low=0, high=len(x), size=batch_size)

            z = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=batch_size)
            a1 = self.discriminator(x[batch])
            a2 = self.generator(z)
            a3 = self.discriminator(a2.detach().numpy())
            discriminator_criterion = torch.mean(a1 - a3)
            self.zero_grad()
            discriminator_criterion.backward()
            with torch.no_grad():  # update the weights
                for param in self.discriminator.parameters():
                    param += learning_rate * param.grad
                if normalise == 's':
                    self.discriminator.normaliseSpectral()
                elif normalise == 'i':
                    self.discriminator.normaliseInfty()

            if num_epoch % 10 == 0:
                print("epoch {}, discriminator_criterion {}, generator_criterion {}".format(num_epoch, discriminator_criterion.item(), generator_criterion.item()))

            num_epoch += 1
