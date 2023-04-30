import numpy as np
import torch
from torch.nn import functional as func
import matplotlib as mpl
import matplotlib.pyplot as plt
print(mpl.get_backend())

import network
from activations import *

start = -3
end = 3
nb_data = 100

X = np.linspace(start, end, nb_data)

#f = np.cos
#f = np.sin
#f = np.exp
#f = np.log
#f = np.arctan
f = np.abs
y = f(X)

# with errors
#X = np.concatenate((X,np.array([1])))
#y = np.concatenate((y,np.array([0])))

r = func.relu
d = deepspline.DeepSpline()
g2 = groupsort.GroupSort(num_units=2)
g4 = groupsort.GroupSort(num_units=4)

nn = network.Net(1, 100, 1, activation=d)
nn.train_(X, y,
              nb_epoch=15000,
              batch_size=30,
              loss_type=2,
              learning_rate=1e-2,
              stopping_loss=1e-3,
              normalise = 's')

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
"""