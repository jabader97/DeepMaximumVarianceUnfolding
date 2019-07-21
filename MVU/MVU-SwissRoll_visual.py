import sklearn.datasets
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import time
from sklearn.neighbors import NearestNeighbors
start_time = time.time()
x, t = sklearn.datasets.make_swiss_roll(300, random_state=0)

# created to make and display short circuiting scenarios

r = 2  # number of final dimensions
m = len(x)  # number of samples
n = len(x[0])  # number of parameters
k = 8  # number of nearest neighbors which forms the neighborhood
eig_tol = 1.0e-10


# squeeze to increase accuracy
initGraph = x.transpose()
initGraph[1] = initGraph[1]/2

x = initGraph.transpose()

# normalize the x matrix
for i in range(0, n):
    col_sum = 0
    for j in range(0, m):
        col_sum += x[j][i]
    mean = col_sum / m
    for j in range(0, m):
        x[j][i] = x[j][i] - mean

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
print('Defining neighborhood')
# for i in range(0, m):
#     row = sd_matrix[i]
#     new_row = list()
#     for j in range(0, m):
#         new_row.append((j, sd_matrix[i][j]))
#     row = sorted(new_row, key=lambda tup: tup[1])
#     for j in range(0, k):
#         neighborhood[row[j][0]][i] = 1
# neighborhood = neighborhood.transpose()
neighbors = NearestNeighbors(n_neighbors=k).fit(x).kneighbors_graph(x).todense()
neighborhood = np.array(neighbors)
neighborhood[20][290] = 1
neighborhood[290][20] = 1

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
prob.solve(verbose=True)
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
print('Final Data: ')
print("--- %s seconds ---" % (time.time() - start_time))


print(final_data[:, 0])
plt.scatter(final_data[:, 0], final_data[:, 1], c=t)
plt.show()