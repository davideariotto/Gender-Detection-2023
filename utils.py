import numpy as np

classes_list = ["Male", "Female"]
features_list = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"]

# Transform 1-dim row vectors to column vectors.
def mcol(v):
    return v.reshape((v.size, 1))

# Transform 1-dim column  vecotrs to row vectors.
def mrow(v):
    return (v.reshape(1, v.size))

# Load data
def load(filename):
    data_list = []
    labels_list = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                # Create a column vector and append it
                data_list.append(mcol(np.array(data[0:-1])))
                labels_list.append(data[-1])
    # Stack horizontally all column vectors to form a matrix
    dataset_matrix = np.hstack(data_list[:])
    # Create a 1-dim array with class labels
    labels_array = np.array(labels_list)
    
    return dataset_matrix, labels_array

# Transform the data into a distribution with 0 mean and 1 variance
def ZNormalization(D):
    mu = D.mean(axis=1)
    sigma = D.std(axis=1)
    normalizedData = (D - mcol(mu)) / mcol(sigma)
    return normalizedData, mu, sigma
