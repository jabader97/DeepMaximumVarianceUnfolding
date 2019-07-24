import torch
import torch.nn as nn
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
import random
start_time = time.time()
torch.manual_seed(2)
random.seed(2)

lbda_svd = .00000000000001  # .00000000000001  # scaling for the singular value decomposition factor
lbda_var = .0045  # .0045  # scaling for the variance factor
size = 3000  # number of points
lr = 10  # learning rate
epoch = 100  # number of times to train the neural network
k = 4  # number of nearest neighbors to preserve distance between
dim_1 = 3  # original dimension of the neural network
dim_2 = 20  # first hidden dimension of the neural network
dim_3 = 10  # second hidden dimension of the neural network
dim_4 = 3  # final dimension of the neural network
final_dim = 2  # number of dimensions to reduce to at the end
squeeze = 2  # amount to squeeze the data by (helps accuracy with fewer points on this data set)


def load_iris(size):
    """
    Function to load the data set and
    :param size: number of points to load
    :return: the data, labels for the data
    """
    data, labels = sklearn.datasets.make_swiss_roll(size, random_state=22)  # seeded random state for consistency
    # transpose to squeeze in the correct direction
    data = data.transpose()
    # squeeze the data to improve accuracy with fewer points for the Swiss Roll data set
    data[1] = data[1] / squeeze
    # return to the original data orientation
    data = data.transpose()
    return data, labels


def normalize(data):
    """
    Function to normalize the data
    :param data: data to be normalized
    :return: normalized data
    """
    m = np.size(data, 0)  # number of data points
    n = np.size(data, 1)  # dimensionality of data points
    for j in range(n):
        # find the average for each column
        col_sum = 0
        for i in range(m):
            col_sum += data[i][j]
        col_sum /= m
        # subtract the average from each value in the column
        for i in range(m):
            data[i][j] -= col_sum
    return data


class Net(nn.Module):
    """
    Neural network to train
    """
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(dim_1, dim_2, bias=True)
        self.f2 = nn.Linear(dim_2, dim_3, bias=True)
        self.f3 = nn.Linear(dim_3, dim_4, bias=True)

    def encode(self, x):
        m = nn.LeakyReLU()
        x = m(self.f(x))
        x = m(self.f2(x))
        x = self.f3(x)
        return x

    def decode(self, x):
        """
        Holder function to be used to return to the original data (like an auto-encoder)
        To be implemented in future work
        :param x: data
        :return: decoded data
        """
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


def train_swiss(epoch, data, net, opti):
    """
    Function to train the network
    :param epoch: number of times to train
    :param data: data to train with
    :param net: neural network to train
    :param opti: optimizer to use during training
    :return:
    """
    # find the distances between each point
    data_distances = pairwise_distances(data)
    data_distances_masked = data_distances * nbr_graph_tensor.float()
    # train the network
    for num in range(epoch):
        global k, lbda
        out = net(data, False)  # run the data through the network
        svd_loss, out = implement_svd(out)  # implement svd and find the L2,1 loss
        output_distances = pairwise_distances(out)  # find the distances of the data modified by svd
        # Multiply the distances between each pair of points with the neighbor mask of the data modified by svd
        output_distances_masked = output_distances * nbr_graph_tensor.float()
        # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
        nbr_diff = torch.abs((output_distances_masked - data_distances_masked))
        nbr_distance = nbr_diff.norm()
        svd_loss *= lbda_svd  # multiply the svd term by its scaling factor
        # calculate variance in all directions multiplied by its scaling factor
        var = 0
        for i in range(out.size()[0]):
            var += lbda_var / out[i].var()
        loss = nbr_distance + svd_loss + var  # loss function is made of all three terms
        opti.zero_grad()
        loss.backward()
        opti.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num + 1, epoch, loss.item()))


def test_swiss(data, net, t):
    """
    Function to test the accuracy of the trained neural network
    :param data: testing data
    :param net: pre-trained neural network
    :param t: labels for the test data
    :return: fully transformed data
    """
    global final_dim
    out = net(data, False)
    u, s, v = torch.svd(out)  # implement svd
    # note: the u returned by this function only includes the top values.
    # u * s will be equivalent due to the zero terms, but will run more efficiently with this implementation.
    top_vals = torch.diag(s)  # create the diagonal matrix from s
    top_vals = top_vals[:, :final_dim]  # cut down s with SVD
    final_data = torch.mm(u, top_vals).detach().numpy()  # u * s
    # plot the final representation
    plt.scatter(final_data[:, 0], final_data[:, 1], c=t, marker='o')
    plt.show()
    return final_data


def implement_svd(data):
    """
    Function to implement svd on the data and find the L2,1 regularization term
    :param data: data to be reduced with SVD
    :return: L2,1 regularization term and transformed matrix
    """
    u, s, v = torch.svd(data)  # implement svd
    # note: the u returned by this function only includes the top values.
    # u * s will be equivalent due to the zero terms, but will run more efficiently with this implementation.
    s = torch.diag(s)  # turn s into a diagonal matrix
    transformed_matrix = torch.mm(u, s)  # u * s
    return l21_reg(s), transformed_matrix  # return the L2,1 regularization term and matrix


def l21_reg(data):
    """
    Function to find the L2,1 regularization term
    :param data: matrix to find the L2,1 regularization term of
    :return: L2,1 regularization term
    """
    m = data.size()[0]  # number of data points
    n = data.size()[1]  # number of dimensions on the data points
    # find L2,1 regularization term
    outer_sum = 0
    for i in range(m):
        inner_sum = 0
        for j in range(n):
            inner_sum += data[i][j] ** 2
        outer_sum += inner_sum ** 0.5
    return outer_sum


def pairwise_distances(x):
    '''
    TODO cite this
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)  # square every element, sum, resize to list
    y = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y)
    return torch.clamp(dist, 0.0, np.inf)


def run_swiss():
    """
    Function to run the network
    :return:
    """
    # training data
    data, t = load_iris(size)
    data = normalize(data)  # normalize the data
    # create the neighborhood graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    nbr_graph = nbrs.kneighbors_graph(data).toarray()
    global nbr_graph_tensor
    nbr_graph_tensor = torch.tensor(nbr_graph)
    # create the data, neural network, and optimizer
    data = torch.from_numpy(data).float()
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3, lr=lr)
    # train net
    train_swiss(epoch, data, net, opti)
    # load testing data
    data, t = load_iris(1000)
    data = normalize(data)  # normalize the testing data
    data = torch.from_numpy(data).float()
    # test the testing data
    test_swiss(data, net, t)


run_swiss()
