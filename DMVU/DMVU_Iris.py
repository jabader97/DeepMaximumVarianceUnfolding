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
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
start_time = time.time()
torch.manual_seed(2)

"""
Hyper-Parameters:
epoch: Number of Epochs for training
size: Size of the training data from Swiss Roll
lbda: Lambda sparsity coefficient for loss
K: Number of neighbors to define local neighborhoods to maintain
lr: Learning Rate
"""
epoch = 10000
size = 150
lbda = 7
k = 50
lr = .0001

"""
Hyperparameters for decent accuracy:
epoch = 10000
size = 150
lbda = 7
k = 50
lr = .0001
"""


def load_iris(size):
    """
    Loads points from the Iris Dataset and returns them in an array and an array of labels.
    
    Input:
        size: Number of points to request to be loaded from the Iris Datset
    Output:
        data: Dataset of points from the Iris Dataset
        labels: Categorization labels for data
    """
    data, labels = sklearn.datasets.load_iris(size)  # 4 dimensional data set to be reduced
    return data, labels


final_in, final_labels = load_iris(1000)


def normalize(data):
    """
    Normalizes data around the origin.
    
    Input:
        data: Array-like to be normalized
    Output:
        data: Normalize dataset
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

    return data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(4, 10, bias=True)
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

def train_iris(epoch, data, net, opti, t):
    '''
    Training the neural net on the training dataset
    
    Input:
        epoch: number of Epochs for training
        data: tensor dataset for processing
        net: DMVU neural net
        opti: Neural Net Optimizer
        t: Labels for dataset
    '''
    data = data.view(data.size(0), -1)
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

        loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())


        opti.zero_grad()
        loss.backward()
        opti.step()

        print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(num + 1, epoch, len(data), loss.item()))



def test_iris(data, net, t):
    '''
    Runs the test dataset through the neural net, and graphs them.

    Input:
        data: Testing dataset to be processed
        net: Neural net
        t: labels for testing dataset
    Output:
        out: The lower-dimensional representation of the testing dataset
    '''
    out = net(data, False).detach().numpy()
    plt.scatter(out[:, 0], out[:, 1], c=t, marker='o')
    plt.show()
    return out

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1) # square every element, sum, resize to list
    print(x_norm.shape)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)




def run_iris():
    # training data
    data, t = load_iris(size)
    data = normalize(data)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    nbr_graph = nbrs.kneighbors_graph(data).toarray()
    global nbr_graph_tensor
    nbr_graph_tensor = torch.tensor(nbr_graph)
    data = torch.from_numpy(data).float()
    net = Net()
    # optimizer
    opti = torch.optim.SGD(net.parameters(), lr=lr)
    # train net
    train_iris(epoch, data, net, opti, t)
    data, t = load_iris(1000)
    data = normalize(data)
    # have k means solution
    kmeans = KMeans(n_clusters=3)
    data = torch.from_numpy(data).float()
    out = test_iris(data, net, t)
    #calculate kmeans clusters, and determine accuracy
    kmeans.fit(out)
    vmeasure = v_measure_score(t, kmeans.labels_)
    return vmeasure

print(run_iris())