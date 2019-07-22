"""
@Author: Bader, J, Nelson, D, Chai-Zhang, T
Created: 22/7/19
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
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

"""
Hyper-Parameters for a decent accuracy:

epoch = 20000
train_batch_size = 400
lbda = 10000
weight_initial = .1
k = 8
lr = .01
squeeze = 1
"""

def load_swiss_data(size):
    """
    Loads points from the Swiss Roll Distrabution and returns them in an array and an array of labels.
    
    Input:
        size: Number of points to request to be generated from the Swiss Roll Distrabution
    Output:
        x: Dataset of points from the Swiss Roll Distrabution
        t: Categorization labels for data
    """
    x, t = sklearn.datasets.make_swiss_roll(size)
    return x, t


def normalize(data):
    """
    Normalizes data around the origin.

    Input:
        data: Array-like to be normalized
    Output:
        data: Normalized data set
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
        self.f = nn.Linear(3, 4)
        self.f2 = nn.Linear(4, 2)
        # self.f2 = nn.Linear(linear_dim1, 2)

    def encode(self, x):
        m = nn.PReLU()
        x = m(self.f(x))
        x = m(self.f2(x))
        return x

    def decode(self, x):
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


def train_dmvu(epoch, data, net, loss_func, opti, t):
    '''
    Training the neural net and produces the final lower dimensional representation
    input:
        epoch: number of Epochs for training
        data: tensor dataset for processing
        net: DMVU neural net
        loss_func: Loss function, should be set to DMVULoss
        opti: Neural Net Optimizer
        t: Labels for dataset
    '''
    for num in range(epoch):
        global k, lbda, lr
        data = data.view(data.size(0), -1)
        out = net(data, False)
        global train_loss
        train_loss = loss_func(out, data) # todo pass in only once
        opti.zero_grad()
        train_loss.backward()
        opti.step()
        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(num + 1, epoch,  len(data), train_loss.item()))
        if num == (epoch - 1):
            out = out.detach().numpy()
            print('Final Test Loss:')
            print(train_loss)
            # Graphing
            plt.scatter(out[:, 0], out[:, 1], c=t, marker='.')
            name = 'loss' + str(train_loss.item()) + 'k' + str(k) + 'lr' + str(lr) + 'weight' + str(weight_initial) + '.png'
            plt.title(name)
            plt.show()
            # plt.savefig(name, bbox_inches='tight')
            # plt.clf()


def initialize_weights(net, weight_initial):
    """
    Sets an initial weight for a given neural net

    Input:
        net: The neural net to have it's initial weights sets
        weight_initial: The value to set the initial weights to
    """
    for z in net.modules():
        if isinstance(z, nn.Linear):
            z.weight.data.normal_(0, weight_initial)


def make_neighborhood(data, k):
    """
    Calculates the nearest neightbor matrix for the points of a given dataset
    
    Input:
        data: The dataset to have it's neightbor matrix calculated.
        k: How many nearest neightbors each point in the matrix has.
    Output:
        Nearest Neighbor Matrix of data with k nearest neighbors
    """
    neighbors = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
    return torch.from_numpy(np.array(neighbors))


def find_first_term(data):
    """
    Calculates pairwise distances between points in the dataset.

    Input:
        data: Array-like dataset to have pairwise distances calculated of.
    Output:
        first_term: Pairwise distances of data
    """
    m_og = data.size()[0]
    data_gram = torch.mm(data, torch.t(data))
    data_diag = torch.diag(data_gram)
    data_diag = torch.diag(data_diag)
    ones = torch.ones((m_og, m_og))
    first_term = torch.mm(data_diag, ones)
    first_term += torch.mm(ones, data_diag)
    first_term -= 2 * data_gram
    return first_term


class DMVULoss(torch.nn.Module):
    """
    Custom loss function for DMVU, designed to incentivize maintaining local distances between neighbors while promoting variance in all other distances. 

    Input:
        latent_space: The lower dimensional representation of the dataset
        data: The original dataset
    Output:
        The loss value
    """
    def __init__(self, neighbors):
        super(DMVULoss, self).__init__()
        self.neighbors = neighbors

    def forward(self, latent_space, data):
        m_ls = latent_space.size()[0]
        #Finds pairwise distances for both original and lower-dimensional representation of dataset.
        term_1 = find_first_term(data)
        term_2 = find_first_term(latent_space)
        #Calculates distances between neighbors only
        term_1 = self.neighbors.double() * term_1.double()
        altered_term_2 = self.neighbors.double() * term_2.double()
        #Finds difference between neighbor distances in original and lower-dimensional representation of dataset.
        term_1 = torch.norm(term_1.sub(altered_term_2))
        latent_ones = torch.ones((m_ls, m_ls)).double()
        #Calculates distances between all non-neighbors in the lower-dimensional representation of the dataset
        holder = latent_ones.sub(self.neighbors)
        term_2 = holder.double() * term_2.double()
        term_2 = lbda / torch.norm(term_2)
        return term_1 + term_2

def run_swiss():
    # training data
    data, t = load_swiss_data(train_batch_size)
    data = normalize(data)
    data = torch.from_numpy(data).float()
    # create net
    net = Net()
    # initialize weights
    initialize_weights(net, weight_initial)
    # make neighbor matrix Q
    neighbors = make_neighborhood(data, k)
    first_term = find_first_term(data)
    # create loss function
    loss_func = DMVULoss(neighbors)
    # optimizer
    opti = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    # train net
    train_dmvu(epoch, data, net, loss_func, opti, t)


run_swiss()
