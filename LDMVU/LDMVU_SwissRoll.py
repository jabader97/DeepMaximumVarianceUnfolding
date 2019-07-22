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


# Hyper parameters
m, n, divisor = 0, 0, 0  # will reset later, m = num samples, n = num dimensions, num_batches = num batches

num_lm = 350  # num landmarks to choose
batch_size = 400  # size of each batch
num_batches = 4  # number of batches
size = (batch_size * num_batches) + num_lm
linear_dim1 = 10  # hidden layer dimension in NN
linear_dim2 = 2  # final dimension of NN
lbda = 200000  # scaling term for variance in loss equation
epoch = 50  # number of times to train network
lm_epoch = 500
squeeze = 4  # amount to squeeze the shape by
set_random = False  # random choice of landmarks vs most connected points
k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 2  # number of landmarks each landmark has
k_other = 4  # number of landmarks each regular points has


class Net(nn.Module):
    """
    Neural Network to train the model. Includes two layers.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(3, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)

    def encode(self, x):
        m = nn.PReLU()
        x = m(self.f(x))
        x = self.f2(x)
        return x

    def decode(self, x):
        """
        In progress:
        To expand the representation back to the original representation (as done in auto-encoders).
        """
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


def normalize(data):
    """
    Function to normalize the data
    :param data: original data
    :return: normalized data
    """
    global squeeze, m, n
    for j in range(n):
        # find the sum for each column
        col_sum = 0
        for i in range(m):
            col_sum += data[i][j]
        col_sum /= m
        # subtract the mean from each point in the column
        for i in range(m):
            data[i][j] -= col_sum
    init_graph = data.transpose()
    init_graph[1] = init_graph[1] / squeeze
    data = init_graph.transpose()
    return data


def load_data(size, num_lm):
    """
    Function to load the data
    :param size: total number of points to get
    :param num_lm: number of points which will be landmarks
    :return: the batch loader, landmark points, labels, batched data without landmark points,
    data organized in graphs, neighborhood graph for the landmarks, original data, original labels,
    neighborhood graphs for non-landmarks
    """
    global divisor, m, n, batch_size, set_random
    # import data
    data, labels = sklearn.datasets.make_swiss_roll(size)
    # data, labels = shuffle(data, labels)
    m = np.size(data, 0)
    n = np.size(data, 1)
    data = normalize(data)
    save_labels = labels
    save_data = data
    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    top_landmarks_idxs = []
    if set_random:
        # select x random points in the data set
        for i in range(num_lm):
            index = random.randint(0, m - i)
            land_marks[i] = data[index]
            data = np.delete(data, index, axis=0)
            labels = np.delete(labels, index, axis=0)
    else:
        # pick the points with the most neighbors
        N = NearestNeighbors(n_neighbors=k_start).fit(data).kneighbors_graph(data).todense()
        N = np.array(N)
        num_connections = N.sum(axis=0).argsort()[::-1]
        top_landmarks_idxs = num_connections[:num_lm]
        land_marks = data[top_landmarks_idxs, :]
        data = np.delete(data, top_landmarks_idxs, axis=0)

    # find the nearest landmarks for the landmarks
    landmark_neighbors = NearestNeighbors(n_neighbors=k_lm).fit(land_marks).kneighbors_graph(land_marks).todense()
    # break data into batches
    divisor = int(size / batch_size)
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    batch_graph = np.zeros((divisor, batch_size + num_lm, batch_size + num_lm))
    for i in range(divisor):
        holder = data[batch_size * i: batch_size * (i + 1)]
        # find the nearest landmarks for the rest of the points
        holder_graph = NearestNeighbors(n_neighbors=k_other).fit(land_marks).kneighbors_graph(holder).todense()
        for j in range(batch_size):  # copy over the holder graph
            for l in range(num_lm):
                if holder_graph[j, l] == 1:
                    batch_graph[i, j, l + batch_size] = 1
                    batch_graph[i, l + batch_size, j] = 1
        for j in range(num_lm):  # copy over landmark neighbors
            for l in range(j, num_lm):
                if landmark_neighbors[j, l] == 1 and j != l:
                    batch_graph[i, j + batch_size, l + batch_size] = 1
                    batch_graph[i, l + batch_size, j + batch_size] = 1
        holder = np.concatenate((holder, land_marks))
        batch_loader[i] = holder
    batch_size += num_lm  # adjust the batch size
    return batch_loader, land_marks, labels, data, batch_graph,top_landmarks_idxs, save_data, save_labels, landmark_neighbors


def train_net(epoch, data, net, opti, batch_graph):
    """
    Method to train the full net
    :param epoch: number of times to train
    :param data: batched data
    :param net: neural net object
    :param opti: optimizer
    :param batch_graph: nearest neighbors graphs
    """
    global divisor, batch_size
    for num in range(epoch):
        for batch_id in range(divisor):
            # train each batch
            batch = torch.from_numpy(data[batch_id]).float()
            batch = batch.view(batch_size, -1)
            batch_distances = pairwise_distances(batch)
            nbr_graph_tensor = torch.from_numpy(batch_graph[batch_id]).float()
            batch_distances_masked = batch_distances * nbr_graph_tensor.float()
            global lbda
            out = net(batch, False)
            output_distances = pairwise_distances(out)
            # Multiply the distances between each pair of points with the neighbor mask
            output_distances_masked = output_distances * nbr_graph_tensor.float()
            # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
            nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
            nbr_distance = nbr_diff.norm()
            loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
            opti.zero_grad()
            loss.backward()
            opti.step()
            print('Epoch: %f, Step: %f, Loss: %.2f' % (num, batch_id + 1, loss.data.cpu().numpy()))


def train_lms(epoch, land_marks, net, opti, landmark_neighbors):
    """
    Method to train the landmarks
    :param epoch: number of times to train
    :param land_marks: landmark points
    :param net: neural network to train
    :param opti: optimizer
    :param landmark_neighbors: neighborhood graph
    """
    for num in range(epoch):
        # train with the landmark points
        global lbda
        batch = torch.from_numpy(land_marks).float().view(num_lm, -1)
        batch_distances = pairwise_distances(batch)
        neighbor_graph = torch.from_numpy(landmark_neighbors).float()
        batch_distances_masked = batch_distances * neighbor_graph.float()
        out = net(batch, False)
        output_distances = pairwise_distances(out)
        output_distances_masked = output_distances * neighbor_graph.float()
        nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
        nbr_distance = nbr_diff.norm()
        loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
        opti.zero_grad()
        loss.backward()
        opti.step()
        print('LM Epoch: %f, Loss: %.2f' % (num, loss.data.cpu().numpy()))


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


def evaluate(data, net, t, landmarks):
    """
    method to evaluate the accuracy of the already trained model
    :param data:
    :param net:
    :param t:
    :param landmarks:
    """
    out = net(torch.from_numpy(data).float(), False)
    out = out.detach().numpy()
    #  Graph the data
    plt.scatter(out[:, 0], out[:, 1], c=t, marker='o')
    if not set_random:
        plt.scatter(out[landmarks, 0], out[landmarks, 1], c='Red', marker='+')
    plt.show()


def run():
    """
    Method to run LDMVU
    """
    global num_lm, lm_epoch
    # load the data
    data_loader, land_marks, labels, data, batch_graph, lm_index, save_data, save_labels, landmark_neighbors = load_data(size, num_lm)
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3)
    # train the landmarks first to spread them out
    train_lms(lm_epoch, land_marks, net, opti, landmark_neighbors)
    # train the rest of the points to fit them around the landmarks
    train_net(epoch, data_loader, net, opti, batch_graph)
    # evaluate the accuracy
    evaluate(save_data, net, save_labels, lm_index)


run()
