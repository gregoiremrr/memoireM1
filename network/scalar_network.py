import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as func

class Net(nn.Module):
    def __init__(self, n_in, n_h, n_out, activation=func.relu):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_h)
        self.fc4 = nn.Linear(n_h, n_h)
        self.fc5 = nn.Linear(n_h, n_h)
        self.fc6 = nn.Linear(n_h, n_out)
        self.act = activation

    def forward(self, x):
        x = np.reshape(x, (len(x), -1))
        x = torch.tensor(x)
        x = x.to(torch.float32)

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

    def normaliseInfty(self):
        n1 = max(abs(self.fc1.weight.data)) #NOOOOO L1 Linf
        n2 = max(torch.sum(abs(self.fc2.weight.data), dim=1))
        n3 = max(torch.sum(abs(self.fc3.weight.data), dim=1))
        n4 = max(torch.sum(abs(self.fc4.weight.data), dim=1))
        n5 = max(torch.sum(abs(self.fc5.weight.data), dim=1))
        n6 = torch.sum(abs(self.fc6.weight.data), dim=1)
        self.fc1.weight.data = self.fc1.weight.data / max(n1, 1)
        self.fc2.weight.data = self.fc2.weight.data / max(n2, 1)
        self.fc3.weight.data = self.fc3.weight.data / max(n3, 1)
        self.fc4.weight.data = self.fc4.weight.data / max(n4, 1)
        self.fc5.weight.data = self.fc5.weight.data / max(n5, 1)
        self.fc6.weight.data = self.fc6.weight.data / max(n6, 1)

    def prod_lip(self):
        n1 = torch.linalg.vector_norm(self.fc1.weight.data)
        n2 = torch.linalg.matrix_norm(self.fc2.weight.data, ord=2)
        n3 = torch.linalg.matrix_norm(self.fc3.weight.data, ord=2)
        n4 = torch.linalg.matrix_norm(self.fc4.weight.data, ord=2)
        n5 = torch.linalg.matrix_norm(self.fc5.weight.data, ord=2)
        n6 = torch.linalg.vector_norm(self.fc6.weight.data)
        prod = n1 * n2 * n3 * n4 * n5 * n6
        return prod

    def train_(self, X, y, nb_epoch, batch_size, loss_type, learning_rate=1e-2, stopping_loss=1e-3, normalise=False):
        loss_l = []
        if loss_type == 1:
            criterion = nn.L1Loss()
        if loss_type == 2:
            criterion = nn.MSELoss()
        rng = np.random.default_rng()
        num_epoch = 0
        loss = 1
        nb_normalise = nb_epoch/3
        while num_epoch <= nb_epoch and (loss > stopping_loss or num_epoch <= nb_normalise+1):

            batch = rng.integers(low=0, high=len(X), size=batch_size)
            hat_y = self.forward(X[batch])  # Forward pass: Compute predicted y by passing x to the model
            y_batch = np.reshape(y[batch], (len(batch), -1))
            y_batch = torch.tensor(y_batch).float()
            loss = criterion(hat_y, y_batch)  # Compute loss
            # Zero gradients, perform a backward pass, and update the weights.
            self.zero_grad()  # re-init the gradients (otherwise they are cumulated)
            loss.backward()  # perform back-propagation
            with torch.no_grad():  # update the weights
                for param in self.parameters():
                    param -= learning_rate * param.grad
                if num_epoch >= nb_normalise:
                    if normalise == 's':
                        self.normaliseSpectral()
                    elif normalise == 'i':
                        self.normaliseInfty()

            # --- END CODE HERE
            loss_l.append(loss.item())

            if num_epoch % 1000 == 0:
                print("epoch {}, loss {}".format(num_epoch, loss.item()))
                print("weight6 = ", torch.linalg.vector_norm(self.fc6.weight.data))
                print("prod = ", self.prod_lip())
            num_epoch += 1
        print("Ended on epoch {} with loss {}".format(num_epoch, loss.item()))