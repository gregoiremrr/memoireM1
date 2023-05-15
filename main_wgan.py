import numpy as np
import torch
from torch.nn import functional as func
import matplotlib as mpl
import matplotlib.pyplot as plt

print(mpl.get_backend())

from network import wgan_network
from activations import *

file = "/Users/gregoiremourre/Desktop/cifar-10-batches-py/data_batch_1"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_batch_1 = unpickle(file)

print(data_batch_1.keys())

labels = data_batch_1[b'labels']
data = data_batch_1[b'data']
# data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

data2 = []

for i in range(len(labels)):
    if labels[i] == 8:
        data2.append(data[i].reshape(3,32,32))

#data2 = np.reshape(data2, (len(data2), -1))
data2 = torch.Tensor(data2).to(torch.float32)
print(data2.shape)

n_in = 3
n_h = 512
n_out = 3
z_dim = 10

netWGAN = wgan_network.WGAN(n_in, n_h, n_out, z_dim)
netWGAN.train_(data2,
               nb_epoch=3000,
               batch_size=20,
               n_critic=10,
               generator_learning_rate=1e-2,
               discriminator_learning_rate=1e-3,
               normalise='sf')

test = np.random.multivariate_normal(np.zeros(z_dim), np.eye(z_dim), size=1)
test = torch.tensor(test).to(torch.float32)
test_ = torch.floor(test)
test_ = netWGAN.generator(test_).to(torch.int32)
test_ = np.reshape(test_.detach().numpy(), (3, 32, 32)).transpose(1, 2, 0)
plt.imshow(test_)
plt.show()

"""
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(data2[i])
plt.show()
"""
