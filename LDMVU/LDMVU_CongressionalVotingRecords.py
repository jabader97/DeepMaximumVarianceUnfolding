import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

import random
start_time = time.time()
torch.manual_seed(2)
random.seed(2)

lm_epoch = 5000  # number of epochs for the landmarks
set_size = 435  # size of a set
batch_size = 70  # size of a batch
test_size = 135  # number of testing points
num_lm = 20  # number of landmarks
num_batches = 4
size = (batch_size * num_batches) + num_lm  # total number of training points
lbda = 1000  # scaling term for variance in loss equation
epoch = 5000  # number of times to train network
k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 5  # number of landmarks each landmark has
k_other = 5  # number of landmarks each regular points has
m = 300  # number of data
n = 16  # number of dimensions
categories = 2


class Net(nn.Module):
    """
    Neural Network to train the model. Includes two layers.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(16, 10, bias=True)
        self.f2 = nn.Linear(10, 2, bias=True)

    def encode(self, x):
        m = nn.LeakyReLU()
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


def original_clean():
    """
    Method to clean the data
    :return: data and labels
    """
    dataset = pd.read_csv('Parliment-1984.csv')
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    for i in range(0, 434):
        if y[i] == 'democrat':
            y[i] = 0
        elif y[i] == 'republican':
            y[i] = 1
    y = y.astype(int)

    for a in range(0, 434):
        for b in range(0, 16):
            if 'y' in X[a][b]:
                X[a][b] = 1
            elif 'n' in X[a][b]:
                X[a][b] = 0

    medians = []
    for x in range(0, 16):
        acceptable = []
        for z in range(0, 434):
            if (X[z][x] == 1) or (X[z][x] == 0):
                acceptable.append(X[z][x])
        med = np.median(acceptable)
        medians.append(int(med))

    for c in range(0, 434):
        for d in range(0, 16):
            if (X[c][d] != 1) and (X[c][d] != 0):
                X[c][d] = medians[d]
    X = X.astype(float)
    X = normalize(X)
    return X, y


def normalize(data):
    """
    Function to normalize the data
    :param data: original data
    :return: normalized data
    """
    row = np.size(data, 0)
    col = np.size(data, 1)
    # find the sum for each column
    for j in range(col):
        col_sum = 0
        for i in range(row):
            col_sum = col_sum + data[i][j]
        col_sum = col_sum / row
        # subtract the mean from each point in the column
        for i in range(row):
            data[i][j] = data[i][j] - col_sum
    return data


def load_data():
    """
    Function to load the data
    :param size: total number of points to get
    :param num_lm: number of points which will be landmarks
    :return: the batch loader, landmark points, labels, batched data without landmark points,
    data organized in graphs, neighborhood graph for the landmarks, original data, original labels,
    neighborhood graphs for non-landmarks
    """
    global batch_size, divisor
    # import data
    data, labels = original_clean()
    test_data = data[300:, :]
    test_labels = labels[300:]
    data = data[:300, :]
    labels = labels[:300]

    # make landmarks with points with most neighbors
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
    # create the full neighborhood graph for each batch
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
    return batch_loader, land_marks, labels, data, batch_graph, top_landmarks_idxs, test_data, test_labels, landmark_neighbors


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
        # train each batch
        for batch_id in range(divisor):
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
            loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())
            opti.zero_grad()
            loss.backward()
            opti.step()


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
        loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())
        opti.zero_grad()
        loss.backward()
        opti.step()


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
    print(time.time() - start_time)
    t = t.astype(float)
    out = out.detach().numpy()
    print('New score metric')
    print("Accuracy with centroid")
    print(score(out, t))
    cmap = colors.ListedColormap(['red','blue'])
    plt.scatter(out[:, 0], out[:, 1], c=t, cmap=cmap, marker='o')
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(out)
    vmeasure = v_measure_score(t, kmeans.labels_)
    print("Accuracy with kmeans")
    print(vmeasure)


def run():
    """
    Method to run LDMVU
    """
    global num_lm
    # load the data
    batch_loader, land_marks, labels, data, batch_graph, lmIndex, test_data, test_labels, landmark_neighbors = load_data()
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3)
    # train the landmarks first
    train_lms(lm_epoch, land_marks, net, opti, landmark_neighbors)
    # train the rest of the data points based off the landmarks
    train_net(epoch, batch_loader, net, opti, batch_graph)
    # evaluate the accuracy of the model
    evaluate(test_data, net, test_labels, lmIndex)


def score(final_data, labels):
    """
    Find the accuracy score for the data
    :param final_data: transformed data
    :param labels: data labels
    :return: accuracy
    :return: accuracy
    """
    m = np.size(final_data, 0)
    # Final Scoring
    eval_arr = np.zeros((10, 2))
    count_arr = np.zeros(10)
    for i in range(0, m):
        color_num = int(labels[i])
        eval_arr[color_num][0] += final_data[i][0]
        eval_arr[color_num][1] += final_data[i][1]
        count_arr[color_num] += 1
    for i in range(0, 10):
        eval_arr[i][0] = eval_arr[i][0] / count_arr[i]
        eval_arr[i][1] = eval_arr[i][1] / count_arr[i]
    count = 0
    for i in range(m):
        min_center = 0
        min_center_dist = (final_data[i][0] - eval_arr[0][0]) ** 2 + (final_data[i][1] - eval_arr[0][1]) ** 2
        for j in range(1, 10):
            if ((final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2) < min_center_dist:
                min_center_dist = (final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2
                min_center = j
        if min_center == labels[i]:
            count += 1
    return count / m

set_size = 435
batch_size = 70
test_size = 135
num_lm = 20
size = (batch_size * 4) + num_lm
lbda = 10000
lm_epoch = 5000
epoch = 5000
k_start = 3
k_lm = 4
k_other = 4
m = 300
n = 16
categories = 2

run()
