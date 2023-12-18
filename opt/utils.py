import numpy as np


def batch_generator(X, y, batch_size):
    """
    :param X: numpy array of shape (num_samples, num_features) representing the input data
    :param y: numpy array of shape (num_samples,) representing the target labels
    :param batch_size: int value specifying the size of each batch
    :return: a generator object that yields batches of input data and corresponding labels

    This method takes in input data X and target labels y, and divides them into batches of size batch_size.
    It shuffles the data randomly and yields each batch as a tuple containing X_batch and y_batch.

    Examples:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        batch_size = 2
        generator = batch_generator(X, y, batch_size)

        # Produces the following batches:
        X_batch, y_batch = next(generator)  # X_batch = [[7, 8]], y_batch = [1]
        X_batch, y_batch = next(generator)  # X_batch = [[1, 2], [3, 4]], y_batch = [0, 1]
        X_batch, y_batch = next(generator)  # X_batch = [[5, 6]], y_batch = [0]

    Note: This method assumes that X and y have the same number of samples (i.e., X.shape[0] == y.shape[0]).
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : min(i + batch_size, num_samples)]
        yield X[batch_indices, :], y[batch_indices]

def estimate_L(X, batch_size):
    """
    Estimate the value of L.

    Parameters:
    :param X: The input data.
    :param batch_size: The size of each batch.

    Returns:
    :return: The estimated value of L.
    """
    b = batch_size
    s = X.shape[0] // b
    L = np.max([np.sum(np.linalg.norm(X[j*b : (j+1)*b])**2) / (4*b) for j in range(s)])
    return L