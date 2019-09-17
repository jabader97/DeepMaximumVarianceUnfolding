import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_trustworthyness(original_data, latent_data, k):
    n = np.size(original_data, 0)
    m = np.size(original_data, 1)
    orig_neighbors = NearestNeighbors(n_neighbors=k).fit(original_data)
    latent_neighbors = NearestNeighbors(n_neighbors=k).fit(latent_data)
    sum = 2 / (n * k * ((2 * n) - (3 * k) - 1))
    for i in range(n):
        for j in range(m):
            if (orig_neighbors == 0) and (latent_neighbors == 1):
                sum += n - k  # todo fix n to rank(i,j)
    sum = 1 - sum
    return sum


def find_continuity(original_data, latent_data, k):
    n = np.size(original_data, 0)
    m = np.size(original_data, 1)
    orig_neighbors = NearestNeighbors(n_neighbors=k).fit(original_data)
    latent_neighbors = NearestNeighbors(n_neighbors=k).fit(latent_data)
    sum = 2 / (n * k * ((2 * n) - (3 * k) - 1))
    for i in range(n):
        for j in range(m):
            if (orig_neighbors == 1) and (latent_neighbors == 0):
                sum += n - k  # todo fix n to rank(i,j)
    sum = 1 - sum
    return sum