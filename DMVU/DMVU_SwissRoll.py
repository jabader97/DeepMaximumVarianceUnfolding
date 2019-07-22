"""
@Author: Bader, J, Nelson, D, Chai-Zhang, T
Created: 19/7/19
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()
torch.manual_seed(2)

"""
Hyper-Parameters:
epoch: Number of Epochs for training
train_batch_size: Size of the training data from Swiss Roll
lbda: Lambda sparsity coefficient for loss
K: Number of neighbors to define local neighborhoods to maintain
squeeze: Coefficient for reducing a specific dimension in normalize(), default to 1 for no change.
lr: Learning Rate
"""

epoch = 20000
train_batch_size = 400
lbda = 10000
k = 8
squeeze = 1
lr = .01


def load_swiss_data(size):
    """
    Loads points from the Swiss Roll distrabution and returns them in an array with an array of labels.

    size: Number of points to request to be generated from the Swiss Roll Distrabution.
    """
    x, t = sklearn.datasets.samples_generator.make_swiss_roll(size, random_state=0) # random_state=0
    return x, t


def normalize(data):
    """
    Normalizes data around the origin.

    data: Array-like to be normalized
    """

    m = np.size(data, 0)
    n = np.size(data, 1)

    for j in range(n):
        col_sum = 0
        for i in range(m):
            col_sum += data[i][j]
        col_sum /= m
        for i in range(m):
            data[i][j] -= col_sum
    initGraph = data.transpose()
    initGraph[1] = initGraph[1] / squeeze
    data = initGraph.transpose()
    return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(3, 10, bias=True)
        self.f2 = nn.Linear(10, 2, bias=True)

    def encode(self, x):
        m = nn.PReLU()
        x = m(self.f(x))
        x = self.f2(x)
        return x

    def decode(self, x):
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


def train_swiss_dmvu(epoch, data, net, opti, t):
    data_distances = pairwise_distances(data)
    data_distances_masked = data_distances * nbr_graph_tensor.float()
    for num in range(epoch):
        global k, lbda
        out = net(data, False)
        output_distances = pairwise_distances(out)
        # Multiply the distances between each pair of points with the neighbor mask
        output_distances_masked = output_distances * nbr_graph_tensor.float()
        # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
        nbr_diff = torch.abs((output_distances_masked - data_distances_masked))

        nbr_distance = nbr_diff.norm()

        loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
        opti.zero_grad()
        loss.backward()
        opti.step()

        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(num + 1, epoch, len(data), loss.item()))
        if num == (epoch - 1):
            out = out.detach().numpy()
            print('Final Test Loss:')
            print(loss)
            # Graphing
            plt.scatter(out[:, 0], out[:, 1], c=t, marker='o')
            name = 'loss' + str(loss.item()) + 'k' + str(k) + '.png'
            plt.title(name)
            print("--- %s seconds ---" % (time.time() - start_time))
            plt.show()


def pairwise_distances(x, y=None):
    '''
    Calculates distances across each element to prevent sparsity across only a single dimension.

    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1) # square every element, sum, resize to list
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    return torch.clamp(dist, 0.0, np.inf)


def run_swiss():
    # training data
    data, t = load_swiss_data(train_batch_size)
    data = normalize(data)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    nbr_graph = nbrs.kneighbors_graph(data).toarray()
    global nbr_graph_tensor
    nbr_graph_tensor = torch.tensor(nbr_graph)
    data = torch.from_numpy(data).float()
    net = Net()
    # optimizer
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3, lr=lr)
    # train net
    train_swiss_dmvu(epoch, data, net, opti, t)


run_swiss()
