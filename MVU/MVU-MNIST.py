import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load the dataset


def generate_mnist(num_points):
    print('test')
    file_name = 'MNIST_'
    file_name = file_name + str(num_points) + '.csv'
    if not os.path.isfile(file_name):
        print('Creating new data file')
        text_holder = np.loadtxt("mnist_train.csv", delimiter=",")
        temp_text_holder = []
        for points in range(0, num_points):
            temp_text_holder.append(text_holder[points])
        text_holder = np.asarray(temp_text_holder)
        np.savetxt(file_name, text_holder, delimiter=" ")

    return file_name

#  Run PCA on the matrix
def pca(file, r):
    data = np.loadtxt(file, delimiter=" ")
    labels = data[:, 0]
    data = np.delete(data, 0, 1)

    #normalize
    data = StandardScaler().fit_transform(data)

    #PCA
    pca = PCA(n_components=r)
    data = pca.fit_transform(data)

    return data, labels

#  MVU function
def mvu(k, r, file):
    x = np.loadtxt(file, delimiter=" ")
    labels = x[:,0]
    x = np.delete(x, 0, 1)

    # m is the number of data points
    m = np.size(x, 0)

    temp = []
    for i in range(0, m):
        temp.append(x[i])
    x = np.asarray(temp)

    n = len(x[0])  # number of parameters

    # normalize the x matrix
    for i in range(0, n):
        col_sum = 0
        for j in range(0, m):
            col_sum += x[j][i]
        mean = col_sum / m
        for j in range(0, m):
            x[j][i] = x[j][i] - mean
    print('Matrix normalized')
    # find the square distance matrix
    sd_matrix = np.zeros((m, m))
    for i in range(0, m):
        for j in range(i + 1, m):
            dist_sum = 0
            for l in range(0, n):
                dist_sum += (x[i][l] - x[j][l]) ** 2
            sd_matrix[i][j] = dist_sum
            sd_matrix[j][i] = dist_sum

    # define the neighborhood
    neighborhood = np.zeros((m, m))
    for i in range(0, m):
        new_row = list()
        for j in range(0, m):
            new_row.append((j, sd_matrix[i][j]))
        row = sorted(new_row, key=lambda tup: tup[1])
        for j in range(0, k):
            neighborhood[row[j][0]][i] = 1
    neighborhood = neighborhood.transpose()

    # code constraints and solve problem
    kernel = cp.Variable((m, m), symmetric=True)
    constraints = [kernel >> 0]
    constraints += [cp.sum(kernel) == 0]
    for i in range(0, m):
        for j in range(0, m):
            if neighborhood[i][j] == 1:
                constraints += [kernel[i][i] + kernel[j][j] - (2*kernel[i][j]) == sd_matrix[i][j]]
    prob = cp.Problem(cp.Maximize(cp.trace(kernel)), constraints)
    prob.solve()
    kernel = kernel.value

    # find all eigenvalues and eigenvectors
    e_vals, e_vecs = np.linalg.eig(kernel)

    # pick the top r to create the principal component matrix
    for i in range(0, m - r):
        e_vecs = np.delete(e_vecs, -1, 1)
        e_vals = np.delete(e_vals, -1)

    # take the singular square roots of the kernel's engenvalues and apply along diagonal
    lbda = np.diag(e_vals ** 0.5)
    # multiply the pc matrix by the x matrix to get the final answer
    final_data = lbda.dot(e_vecs.T).T
    return final_data, labels


def graph(final_data, labels, path):
    m = np.size(final_data, 0)
    cmap = colors.ListedColormap(['red','orange','yellow','green','cyan','blue','purple','pink','magenta','brown'])
    plt.scatter(final_data[:,0], final_data[:,1], c=labels[0:m], cmap=cmap, marker='.', s=100, alpha=0.45)
    plt.savefig(path)



def score(final_data, labels):
    m = np.size(final_data, 0)
    #Final Scoring
    eval_arr = np.zeros((10,2))
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
            if ((final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2) < min_center_dist :
                min_center_dist = (final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2
                min_center = j
        if min_center == labels[i]:
            count += 1
    return count/m


points_total = 200
num_neighbors = 4
dimensions_final_pca = 100
dimensions_final_mvu = 199


fname = generate_mnist(points_total)
print('Finished generating points')


max_accuracy = 0

pca_data_set, num_labels = mvu(4, dimensions_final_mvu, fname)
accuracy_score = score(pca_data_set, num_labels)
print(accuracy_score)

#  Graphing module
graph(pca_data_set, num_labels, 'PCA_MNIST.png')
print('All accuracy scores are for:')
print(points_total)