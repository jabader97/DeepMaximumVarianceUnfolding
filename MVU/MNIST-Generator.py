import numpy as np

# used to generate MNIST data for MVU, download a MNIST file and have it in the same file
num_points = 10000
text_holder = np.loadtxt("mnist_train.csv", delimiter=",")
temp_text_holder = []
for points in range(10000, 12001):
    temp_text_holder.append(text_holder[points])
text_holder = np.asarray(temp_text_holder)
file_name = 'MNIST_Test2000'
file_name = file_name + str(num_points)
file_name += '.csv'
np.savetxt(file_name, text_holder, delimiter=" ")