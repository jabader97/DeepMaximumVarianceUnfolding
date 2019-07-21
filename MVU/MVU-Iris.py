import sklearn.datasets
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
start_time = time.time()

# loads data set
def load_iris(size):
    data, labels = sklearn.datasets.load_iris(size)  # 4 dimensional data set to be reduced to
    return data, labels

size = 150

# get data and labels from data set
x, t = load_iris(size)



r = 2  # number of final dimensions
m = len(x)  # number of samples
n = len(x[0])  # number of parameters
k = 50 # number of nearest neighbors which forms the neighborhood
eig_tol = 1.0e-10


m = np.size(x, 0)  # rows
n = np.size(x, 1)  # columns

#  normalize the matrix
for j in range(n):
    col_sum = 0
    for i in range(m):
        col_sum += x[i][j]
    col_sum /= m
    for i in range(m):
        x[i][j] -= col_sum

#  find the square distance matrix
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
print('Defining neighborhood')
for i in range(0, m):
    row = sd_matrix[i]
    new_row = list()
    for j in range(0, m):
        new_row.append((j, sd_matrix[i][j]))
    row = sorted(new_row, key=lambda tup: tup[1])
    for j in range(0, k):
        neighborhood[row[j][0]][i] = 1
neighborhood = neighborhood.transpose()
print(neighborhood)


# code constraints and solve problem
kernel = cp.Variable((m, m), symmetric=True)
constraints = [kernel >> 0]
constraints += [cp.sum(kernel) == 0]
for i in range(0, m):
    print('Assigning constraints')
    for j in range(0, m):
        if neighborhood[i][j] == 1:
            constraints += [kernel[i][i] + kernel[j][j] - (2*kernel[i][j]) == sd_matrix[i][j]]

print('Solving the problem')
prob = cp.Problem(cp.Maximize(cp.trace(kernel)), constraints)
prob.solve(verbose=True)  # verbose will make it print each iteration, could slow down the program slightly
kernel = kernel.value
print(kernel.shape)
# find all eigenvalues and eigenvectors
e_vals, e_vecs = np.linalg.eig(kernel)

# pick the top r to create the principal component matrix
print(n)
for i in range(0, m - r):
    e_vecs = np.delete(e_vecs, -1, 1)
    e_vals = np.delete(e_vals, -1)


# take the singular square roots of the kernel's engenvalues and apply along diagonal
lbda = np.diag(e_vals ** 0.5)
# multiply the pc matrix by the x matrix to get the final answer
final_data = lbda.dot(e_vecs.T).T
# print("--- %s seconds ---" % (time.time() - start_time))
print('----Final time----')
print(time.time()-start_time)

# graphing module
plt.scatter(final_data[:, 0], final_data[:, 1], c=t)
plt.show()

# scoring module
kmeans = KMeans(n_clusters=3)
kmeans.fit(final_data)
vmeasure = v_measure_score(t, kmeans.labels_)
print(vmeasure)