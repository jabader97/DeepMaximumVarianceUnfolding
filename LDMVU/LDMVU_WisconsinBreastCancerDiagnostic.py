import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import random
start_time = time.time()
torch.manual_seed(2)
random.seed(2)


# Hyper parameters
m, n, divisor = 0, 0, 0  # will reset later, m = num samples, n = num dimensions, num_batches = num batches

num_lm = 30  # num landmarks to choose
batch_size = 60  # size of each batch
size = 560  # num batches
linear_dim0 = 30  # total num samples
linear_dim1 = 100  # hidden layer dimension in NN
linear_dim2 = 8  # hidden layer dimension in NN
lbda = 90000  # scaling term for variance in loss equation
epoch = 500  # number of times to train network
squeeze = 2  # amount to squeeze the shape by
set_random = False  # random choice of landmarks vs most connected points
temp_subset = num_lm + (batch_size * 5)
trainsize = 200

k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 3  # number of landmarks each landmark has
k_other = 5  # number of landmarks each regular points has


def normalize(data):
    """
    Method to normalize the data
    :param data:
    :return:
    """
    row = np.size(data, 0)
    col = np.size(data, 1)

    # find the column sums
    for j in range(col):
        col_sum = 0
        for i in range(row):
            col_sum = col_sum + data[i][j]
        col_sum = col_sum / row
        # subtract the mean from each column
        for i in range(row):
            data[i][j] = data[i][j] - col_sum
    return data


class Net(nn.Module):
    """
    Neural Network to train the model. Includes two layers.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(linear_dim0, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)

    def encode(self, x):
        p = nn.LeakyReLU()
        x = p(self.f(x))
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


# load data set, select land marks at random and remove from data set, return a train_loader
def load_data(size, num_lm):
    # load the data
    global divisor, m, n, batch_size, set_random, trainsize, linear_dim0
    both_dataset = np.genfromtxt("wdbc.data", dtype=np.float, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                                                      12, 13, 14, 15, 16, 17, 18, 19,
                                                                                      20, 21, 22, 23, 24, 25, 26, 27,
                                                                                      28, 29, 30, 31), encoding=None)
    both_labels = np.genfromtxt("wdbc.data", dtype=None, delimiter=',', usecols=(1), encoding=None)
    tempAll_labels = np.zeros(len(both_labels)) 
    for i in range(len(both_labels)):
        if both_labels[i] == 'B':
            tempAll_labels[i] = 0
        else:
            tempAll_labels[i] = 1
    size = temp_subset
    m = size
    n = linear_dim0
    test_dataset = both_dataset[temp_subset:]
    train_dataset = both_dataset[:temp_subset]
    test_labels = torch.from_numpy(tempAll_labels[temp_subset:])
    train_labels = torch.from_numpy(tempAll_labels[:temp_subset])

    # normalize
    temp_data = normalize(train_dataset)
    temp_data = torch.from_numpy(temp_data)
    temp_labels = train_labels

    data = np.zeros((size, n))
    for i in range(size):
        data[i] = temp_data[i].view(-1, n)

    temp_data = data

    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    if set_random:
        for i in range(num_lm):
            # select x random points in the data set
            index = random.randint(0, size - i)
            land_marks[i] = temp_data[index]
            temp_data = np.delete(temp_data, index, axis=0)
            temp_labels = np.delete(temp_labels, index, axis=0)
    else:
        # pick the points with the most neighbors
        top_landmarks_idxs = np.zeros(num_lm, dtype=np.int32)
        num_each = int(num_lm / 10)

        for i in range(10):
            count = 0
            for j in range(m):
                if temp_labels[j] == i and count < num_each:
                    index = i * num_each + count
                    count += 1
                    top_landmarks_idxs[index] = j
        land_marks = temp_data[top_landmarks_idxs, :]
        temp_data = np.delete(temp_data, top_landmarks_idxs, axis=0)

    # find the nearest landmarks for the landmarks
    landmark_neighbors = NearestNeighbors(n_neighbors=k_lm).fit(land_marks).kneighbors_graph(land_marks).todense()
    # break data into batches
    divisor = int(size / batch_size)
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    batch_graph = np.zeros((divisor, batch_size + num_lm, batch_size + num_lm))
    # create the full neighborhood graph for each batch
    for i in range(divisor):
        holder = temp_data[batch_size * i: batch_size * (i + 1)]
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
    batch_size += num_lm
    return batch_loader, temp_data, batch_graph, landmark_neighbors, test_dataset, land_marks, test_labels, train_labels


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
            loss = (1 / lbda) * nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())
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
        loss = (1 / lbda) * nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())
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


def evaluate(test_loader, net, num_points, labels):
    """
    method to evaluate the accuracy of the already trained model
    :param data:
    :param net:
    :param t:
    :param landmarks:
    """
    global linear_dim0
    n = linear_dim0
    correct = 0
    total = 0
    x = 0
    final_score = 0
    # check each image
    for images in test_loader:
        x += 1
        images = images.reshape(-1, n)
        out = net(images.float(), False)
        none, predicted = torch.max(out.data, 1)
        print(predicted)
        mal = max(predicted.numpy())
        print(mal)
        for i in range(len(predicted)):
            if predicted[i] == mal:
                predicted[i] = 1
            else:
                predicted[i] = 0
        print(predicted)
        total += len(labels)
        correct += (predicted == labels.long()).sum().item()
        print('%f,%f,%f,%f,%f,%f,%f,%f,%f' % ((100 * correct / total), num_lm, batch_size , lbda, k_start , k_lm,
                                              k_other, linear_dim1, linear_dim2))
        out = out.detach().numpy()
        x += 1
        final_score = 100 * correct / total
    print(x)
    return final_score


def run():
    """
    Method to run LDMVU
    """
    num_points = 10000
    global num_lm
    # load the data
    data_loader, data, batch_graph, landmark_neighbors, test_dataset, land_marks, test_labels, train_labels = load_data(size, num_lm)
    net = Net()
    net = net.float()
    opti = torch.optim.SGD(net.parameters(), lr=.0001, momentum=.1)
    for i in net.modules():
        if isinstance(i, nn.Linear):
            i.weight.data.normal_(0, .1)
    # train the landmarks first to spread them out
    train_lms(epoch, land_marks, net, opti, landmark_neighbors)
    # train the rest of the points to fit them around the landmarks
    train_net(epoch, data_loader, net, opti, batch_graph)
    # change test_loader batch size to increase number of points tested on
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=num_points, shuffle=False)
    return evaluate(test_loader, net, num_points, test_labels)


run()
