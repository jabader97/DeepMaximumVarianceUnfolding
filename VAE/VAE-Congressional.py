import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# changed configuration to this instead of argparse for easier interaction
SEED = 1
LOG_INTERVAL = 10
EPOCHS = 100
size = 436
squeeze = 5
orig_dim = 16
low_dim = 2
high_dim = 15  # 15

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 11
data = 0  # will be set later

torch.manual_seed(SEED)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(orig_dim, high_dim)
        self.fc2 = nn.Linear(high_dim, low_dim)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.LeakyReLU()

        self.fc21 = nn.Linear(low_dim, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(low_dim, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(ZDIMS, low_dim)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(low_dim, high_dim)
        self.fc5 = nn.Linear(high_dim, orig_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
            variance units one for each latent dimension

        """

        # h1 is [128, 400]
        h1 = self.relu(self.fc1(x))  # type: Variable
        h1 = self.relu(self.fc2(h1))
        return h1, self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        h3 = self.fc4(h3)
        return self.sigmoid(self.fc5(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        latent, mu, logvar = self.encode(x.view(-1, 16))
        z = self.reparameterize(mu, logvar)
        return latent, self.decode(z), mu, logvar


model = VAE()


def loss_function(recon_x, x, mu, logvar) -> Variable:
    # how well do input x and output recon_x agree?
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 16))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= orig_dim

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD


# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train():
    # toggle model to train mode
    global data
    model.train()
    train_loss = 0
    optimizer.zero_grad()

    # push whole batch of data through VAE.forward() to get recon_loss
    latent, recon_batch, mu, logvar = model(data)
    # calculate scalar loss
    loss = loss_function(recon_batch, data, mu, logvar)
    # calculate the gradient of the loss w.r.t. the graph leaves
    # i.e. input variables -- by the power of pytorch!
    loss.backward()
    train_loss += loss.data
    optimizer.step()
    return latent.detach().numpy()


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
    X = torch.from_numpy(normalize(X)).float()
    return X, y


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


data, labels = original_clean()
for epoch in range(1, EPOCHS + 1):
    out = train()
plt.scatter(out[:, 0], out[:, 1], c=labels, marker='o')
plt.show()
