import numpy as np
from torch.nn import functional as func
import matplotlib as mpl
import matplotlib.pyplot as plt

print(mpl.get_backend())

from network import wgan_network
from activations import *

r = func.relu
d = deepspline.DeepSpline()
g2 = groupsort.GroupSort(num_units=2)
g4 = groupsort.GroupSort(num_units=4)

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
#data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

data2 = []

for i in range(len(labels)):
    if labels[i] == 8:
        data2.append(data[i])

data2 = np.array(data2)
"""
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(data2[i])
plt.show()
"""

n_in = 3072
n_h = 200

netWGAN = wgan_network.WGAN(r, n_in, n_h)
netWGAN.train2_(data2,
               nb_epoch=1000,
               batch_size=40,
               n_critic=10,
               learning_rate=1e-5,
               normalise='s')

print(netWGAN.generator(np.random.multivariate_normal(np.zeros(10), np.eye(10), size=1)))