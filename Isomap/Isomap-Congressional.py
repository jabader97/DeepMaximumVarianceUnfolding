from sklearn.manifold import Isomap
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score


squeeze = 1
random.seed(2)
size = 435


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


x, y = original_clean()
model = Isomap(n_components=size, n_neighbors=45)  # 45 = 48%
out = model.fit_transform(x)
out = out[:, 0:2]
plt.scatter(out[:, 0], out[:, 1], c=y, marker='o')
plt.show()
model = DBSCAN()
predicted = model.fit_predict(out)
score = v_measure_score(predicted, y)
print(score)
