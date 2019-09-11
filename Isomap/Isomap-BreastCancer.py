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
    # load the data
    dataset = np.genfromtxt("wdbc.data", dtype=np.float, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                                                      12, 13, 14, 15, 16, 17, 18, 19,
                                                                                      20, 21, 22, 23, 24, 25, 26, 27,
                                                                                      28, 29, 30, 31), encoding=None)
    labels = np.genfromtxt("wdbc.data", dtype=None, delimiter=',', usecols=(1), encoding=None)
    temp_labels = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] == 'B':
            temp_labels[i] = 0
        else:
            temp_labels[i] = 1
    # normalize
    temp_data = normalize(dataset)
    return temp_data, temp_labels


x, y = original_clean()
model = Isomap(n_components=size, n_neighbors=30)
out = model.fit_transform(x)
out = out[:, 0:2]
plt.scatter(out[:, 0], out[:, 1], c=y, marker='o')
plt.show()
model_2 = DBSCAN()
predicted = model_2.fit_predict(out)
score = v_measure_score(predicted, y)
print(score)
