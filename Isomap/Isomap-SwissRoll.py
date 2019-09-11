from sklearn.manifold import Isomap
import sklearn.datasets
import random
import matplotlib.pyplot as plt


squeeze = 1
size = 3000
random.seed(2)


def load_iris(size):
    """
    Function to load the data set and
    :param size: number of points to load
    :return: the data, labels for the data
    """
    data, labels = sklearn.datasets.make_swiss_roll(size, random_state=22)  # seeded random state for consistency
    # transpose to squeeze in the correct direction
    data = data.transpose()
    # squeeze the data to improve accuracy with fewer points for the Swiss Roll data set
    data[1] = data[1] / squeeze
    # return to the original data orientation
    data = data.transpose()
    return data, labels


x, y = load_iris(size)
model = Isomap(n_components=size, n_neighbors=4)
out = model.fit_transform(x)
plt.scatter(out[:, 0], out[:, 1], c=y, marker='o')
plt.show()
