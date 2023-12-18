import numpy as np


def batch_generator(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : min(i + batch_size, num_samples)]
        yield X[batch_indices, :], y[batch_indices]

def estimate_L(X, batch_size):
    b = batch_size
    s = X.shape[0] // b
    L = np.max([np.sum(np.linalg.norm(X[j*b : (j+1)*b])**2) / (4*b) for j in range(s)])
    return L