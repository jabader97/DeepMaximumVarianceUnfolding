import torch
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import random
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score
start_time = time.time()
torch.manual_seed(2)
random.seed(2)


# Hyper parameters
num_original_dim = 16  # number of dimensions in the original data
lbda_svd = .000001  # scaling for the singular value decomposition factor .00000000000001
lbda_var = .0045  # scaling for the variance factor .0045

num_lm = 100  # number of landmarks, must be a multiple of ten
batch_size = 100  # number of points in each batch
num_batches = 2  # number of batches to use
test_size = 60  # number of points to use for testing
size = num_lm + (batch_size * num_batches)  # total number of points needed

lr = .1  # learning rate .0001
linear_dim1 = 10  # dimension of the first hidden layer
linear_dim2 = 32  # dimension of the second hidden layer
linear_dim3 = 32   # dimension of the third hidden layer
linear_dim4 = 10  # dimension of the fourth hidden layer
linear_dim5 = num_original_dim  # dimension of the final layer
epoch = 5  # number of times to train the network
set_random = False  # if true, choose landmarks randomly
temp_subset = num_lm + (batch_size * 5)  # total number of points to acquire
final_dim = 0  # number of dimensions to reduce to at the end

k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 5  # number of landmarks each landmark has
k_other = 10  # number of landmarks each regular points has

m, n = size, 16  # number of samples, number of dimensions/parameters


def normalize(data):
    """
    Function to normalize the data
    :param data: data to be normalized
    :return: normalized data
    """
    row = np.size(data, 0)  # number of data points
    col = np.size(data, 1)  # dimensionality of data points
    for j in range(col):
        # find the average for each column
        col_sum = 0
        for i in range(row):
            col_sum = col_sum + data[i][j]
        col_sum = col_sum / row
        # subtract the average from each value in the column
        for i in range(row):
            data[i][j] = data[i][j] - col_sum
    return data


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


class Net(nn.Module):
    """
    Neural network to train
    """
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(num_original_dim, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)
        self.f3 = nn.Linear(linear_dim2, linear_dim3, bias=True)
        self.f4 = nn.Linear(linear_dim3, linear_dim4, bias=True)
        self.f5 = nn.Linear(linear_dim4, linear_dim5, bias=True)

    def encode(self, x):
        p = nn.LeakyReLU()
        x = p(self.f(x))
        x = p(self.f2(x))
        x = p(self.f3(x))
        x = p(self.f4(x))
        x = self.f5(x)
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


def load_data():
    """
    Function to load the data
    :param size: total number of points to get
    :param num_lm: number of points which will be landmarks
    :return: the batch loader, landmark points, labels, batched data without landmark points,
    data organized in graphs, neighborhood graph for the landmarks, original data, original labels,
    neighborhood graphs for non-landmarks
    """
    global batch_size, num_batches
    # import data
    data, labels = original_clean()
    test_data = data[:test_size, :]
    test_labels = labels[:test_size]

    data = data[test_size:, :]

    # make landmarks with points with most neighbors
    N = NearestNeighbors(n_neighbors=k_start).fit(data).kneighbors_graph(data).todense()
    N = np.array(N)
    num_connections = N.sum(axis=0).argsort()[::-1]  # see how many neighbors each point has
    top_landmarks_idxs = num_connections[:num_lm]  # sort in descending order
    land_marks = data[top_landmarks_idxs, :]  # pick the top ones
    data = np.delete(data, top_landmarks_idxs, axis=0)  # delete the landmarks
    # find the nearest landmarks for the landmarks
    landmark_neighbors = NearestNeighbors(n_neighbors=k_lm).fit(land_marks).kneighbors_graph(land_marks).todense()
    # break data into batches, create empty holders
    batch_loader = np.zeros((num_batches, batch_size + num_lm, n))
    batch_graph = np.zeros((num_batches, batch_size + num_lm, batch_size + num_lm))
    # create the full neighborhood graph for each batch
    for i in range(num_batches):
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
    return batch_loader, data, batch_graph, landmark_neighbors, test_data, test_labels, land_marks


def train_net(epoch, data, net, opti, batch_graph):
    """
    Function to train the network
    :param epoch: number of times to train
    :param data: data to train with
    :param net: neural network to train
    :param opti: optimizer to use during training
    :return:
    """
    global num_batches, batch_size
    # train the network
    for num in range(epoch):
        # run each batch through each round
        for batch_id in range(num_batches):
            # calculate the neighborhood for the graph
            batch = torch.from_numpy(data[batch_id]).float()
            batch = batch.view(batch_size, -1)
            batch_distances = pairwise_distances(batch)
            nbr_graph_tensor = torch.from_numpy(batch_graph[batch_id]).float()
            batch_distances_masked = batch_distances * nbr_graph_tensor.float()
            global lbda
            out = net(batch, False)  # run the batch through the network
            svd_loss, out = implement_svd(out)  # calculate the SVD L2,1 loss and SVD representation
            output_distances = pairwise_distances(out)
            # Multiply the distances between each pair of points with the neighbor mask
            output_distances_masked = output_distances * nbr_graph_tensor.float()
            # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
            nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
            nbr_distance = nbr_diff.norm()
            svd_loss *= lbda_svd  # multiply SVD loss by its scaling factor
            # find variance in all directions
            var = 0
            for i in range(out.size()[0]):
                var += lbda_var / out[i].var()
            loss = nbr_distance + svd_loss + var  # loss contains all three terms
            opti.zero_grad()
            loss.backward()
            opti.step()
            print('Epoch: %f, Step: %f, Loss: %.2f' % (num, batch_id + 1, loss.data.cpu().numpy()))

    # find the ideal number of dimensions
    global final_dim
    batch = torch.from_numpy(data[0]).float()
    batch = batch.view(batch_size, -1)
    out = net(batch, False)
    u, s, v = torch.svd(out)
    final_dim = calc_dim(s)


def train_lms(epoch, land_marks, net, opti, landmark_neighbors):
    """
    Function to train the landmarks to spread them out
    :param epoch: number of times to train the network
    :param land_marks: points to use for distance calculations
    :param net: neural network to train
    :param opti: optimizer to use for the network
    :param landmark_neighbors: nearest neighbors of the landmarks
    :return:
    """
    # find the neighborhood graphs for the landmarks
    batch = torch.from_numpy(land_marks).float().view(num_lm, -1)
    batch_distances = pairwise_distances(batch)
    neighbor_graph = torch.from_numpy(landmark_neighbors).float()
    batch_distances_masked = batch_distances * neighbor_graph.float()
    # train the network
    for num in range(epoch):
        global lbda
        print(batch.shape)
        out = net(batch, False)  # put the data through the network
        svd_loss, out = implement_svd(out)  # find SVD translation and L2,1 regularization
        output_distances = pairwise_distances(out)
        # Multiply the distances between each pair of points with the neighbor mask
        output_distances_masked = output_distances * neighbor_graph.float()
        # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
        nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
        nbr_distance = nbr_diff.norm()
        svd_loss *= lbda_svd  # multiply SVD term by its scaling factor
        # calculate variance in all direction
        var = 0
        for i in range(out.size()[0]):
            var += lbda_var / out[i].var()
        loss = nbr_distance + svd_loss + var  # loss includes all three terms
        opti.zero_grad()
        loss.backward()
        opti.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(num + 1, epoch, loss.item()))


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


def evaluate(test_dataset, test_labels, net):
    """
    Function to evaluate the accuracy of the model
    :param test_dataset: test data
    :param test_labels: lables for the test data
    :param net: pre-trained data
    :return: accuracy score
    """
    rep = torch.zeros((test_dataset.shape[0], final_dim))
    # find each lower dimensional representation
    d = torch.from_numpy(test_dataset)
    net = net.double()
    out = net(d, False)  # put through the network
    u, s, v = torch.svd(out)  # implement SVD
    top_vals = torch.diag(s)  # create the diagonal matrix from s
    top_vals = top_vals[:, :final_dim]  # cut down s with SVD
    rep = torch.mm(u, top_vals)
    # evaluate the accuracy of the representation
    rep = rep.detach().numpy()
    model = DBSCAN()
    predicted = model.fit_predict(rep)
    score = v_measure_score(predicted, test_labels)
    return rep, score


def calc_dim(s):
    """
    Function to calculate the number of dimensions which would equal 90% of the data information
    :param s: S vector from SVD
    :return: suggested number of dimensions to use
    """
    s = s.detach().numpy()
    dim = 0
    # calculate how much 90% would be
    s_square = [i ** 2 for i in s]
    sum_square = sum(s_square)
    goal = .9 * sum_square
    # find 90%
    count = 0
    while count < goal:
        count += s_square[dim]
        dim += 1
    return dim  # return this many dimensions


def run():
    global num_lm
    # load the data
    data_loader, data, batch_graph, landmark_neighbors, test_dataset, test_labels, land_marks = load_data()
    # create the network and initialize the weights
    net = Net()
    opti = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    train_lms(epoch, land_marks, net, opti, landmark_neighbors)  # train the landmarks only first
    train_net(epoch, data_loader, net, opti, batch_graph)  # train all other points to fit around the landmarks
    # load test data
    return evaluate(test_dataset, test_labels, net)  # evaluate model accuracy


rep, score = run()
print(score)
