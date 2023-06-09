import numpy as np
import torch
from torch.nn import functional as func
import matplotlib as mpl
import matplotlib.pyplot as plt
print(mpl.get_backend())

from network import scalar_network
from activations import *

start = -3
end = 3
nb_data = 100

X = np.linspace(start, end, nb_data)
X = np.reshape(X, (len(X), -1))
X = torch.tensor(X)
X = X.to(torch.float32)

#f = np.cos
f = np.sin
#f = np.exp
#f = np.log
#f = np.arctan
#f = np.abs
y = f(X)*2
y = np.reshape(y, (len(y), -1))
y = torch.tensor(y)

# with errors
#X = torch.cat((X, torch.tensor([[0]])), 0)
#y = torch.cat((y, torch.tensor([[2]])), 0)

r = func.relu
d = deepspline.DeepSpline()
g2 = groupsort.GroupSort(num_units=2)
g4 = groupsort.GroupSort(num_units=4)

nn = scalar_network.Net(1, 100, 1, activation=d)
nn.train_(X, y,
              nb_epoch=15000,
              batch_size=30,
              loss_type=2,
              learning_rate=1e-2,
              stopping_loss=1e-4,
              normalise='s')

fig, ax = plt.subplots()
ax.plot(X, y, '+', color='r', label="Train set")
ax.plot(X, nn(X).detach().numpy(), 'x', color='g', label="Neural network")
ax.axis('equal')
leg = ax.legend()
plt.show()

"""
plt.plot(np.linspace(-1,2,100), d(torch.tensor(np.linspace(-1,2,100))).detach().numpy())
plt.show()

print("coeff1 = ", d(torch.tensor(0))-d(torch.tensor(-1)))
print("coeff2 = ", d(torch.tensor(1))-d(torch.tensor(0)))
print("coeff2 = ", d(torch.tensor(2))-d(torch.tensor(1)))

fig2, ax2 = plt.subplots()
ax2.plot(X, nn(X).detach().numpy()-y, 'x', color='g', label="Neural network")
plt.show()
"""