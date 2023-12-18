import numpy as np

def local_model(Xtrain, Ytrain, Xtest, Ytest, lambda_val, opts):
    """
    :param Xtrain: numpy array, training data features
    :param Ytrain: numpy array, training data labels
    :param Xtest: numpy array, testing data features
    :param Ytest: numpy array, testing data labels
    :param lambda_val: float, regularization parameter
    :param opts: dictionary, additional options for the method
    :return: float, average error of the model

    This method trains a local model on the given training data and predicts labels for the testing data.
    The method supports both regression and classification models, depending on the value of the 'obj' key in the opts dictionary.

    For regression models ('obj' key is set to 'R'), the method calculates the predicted values and calculates the average error.
    If the 'avg' key is True in the opts dictionary, the method calculates the error for each training instance and returns the average error.
    If the 'avg' key is False, the method calculates the error for each training instance and concatenates the predicted values and actual labels to calculate the average error.

    For classification models, the method uses the simple_svm function to train the model on each training instance and predicts labels for the testing data.
    If the 'avg' key is True in the opts dictionary, the method calculates the error for each training instance and returns the average error.
    If the 'avg' key is False, the method concatenates the predicted labels and actual labels to calculate the average error.

    Returns the average error of the model.

    """
    m = len(Xtrain)
    d = Xtrain[0].shape[1]

    if opts['obj'] == 'R':
            if opts['avg']:
                errs = np.zeros(m)
                for t in range(m):
                    wt = np.linalg.inv(np.transpose(Xtrain[t]) @ Xtrain[t] + lambda_val * np.eye(d)) @ np.transpose(Xtrain[t]) @ Ytrain[t]
                    errs[t] = np.sqrt(np.mean(np.square(Ytest[t] - Xtest[t] @ wt)))
                err = np.mean(errs)
            else:
                Y_hat = [None]*m
                for t in range(m):
                    wt = np.linalg.inv(np.transpose(Xtrain[t]) @ Xtrain[t] + lambda_val * np.eye(d)) @ np.transpose(Xtrain[t]) @ Ytrain[t]
                    Y_hat[t] = Xtest[t] @ wt
                Y = np.concatenate(Ytest, axis=0)
                Y_hat = np.concatenate(Y_hat, axis=0)

                err = np.sqrt(np.mean(np.square(Y - Y_hat)))
    else:  # classification
        Y_hat = [None] * m
    for t in range(m):
        wt = simple_svm(Xtrain[t], Ytrain[t], lambda_val, opts)

        Y_hat[t] = np.sign(np.dot(Xtest[t], wt))

    if opts['avg']:
        errs = np.zeros(m)
        for t in range (m):
            errs[t] = np.mean(Ytest[t] != Y_hat[t])
        err = np.mean(errs)
    else:
        Y = np.concatenate(Ytest)

        Y_hat = np.concatenate(Y_hat)

        err = np.mean(Y != Y_hat)

    return err



def simple_svm(X, y, lambda_val, opts):
    """
    :param X: numpy array representing the input data with shape (n, d)
    :param y: numpy array representing the target labels with shape (n,)
    :param lambda_val: regularization parameter
    :param opts: dictionary containing additional options
    :return: numpy array representing the SVM weights with shape (d,)

    """
    n, d = X.shape
    w = np.zeros((d, ))
    alpha = np.zeros((n, ))
    primal_old = 0

    for iter in range(opts['max_sdca_iters']):
        # update coordinates cyclically
        for i in range(n):
            # get current variables
            alpha_old = alpha[i]
            curr_x = X[i, :]
            curr_y = y[i]

            # calculate update
            update = curr_y * np.dot(curr_x,w)

            grad = lambda_val * n * (1.0 - update) / (np.dot(curr_x, curr_x) + alpha_old * curr_y)
            # apply update
            alpha[i]  = curr_y * np.mean(np.maximum(0, np.minimum(1.0, grad)))

            w = w + ((alpha[i] - alpha_old) * curr_x.T * (1.0 / (lambda_val * n)))




        # # break if less than tol
        # preds = y @ (np.matmul(X,w))
        # primal_new = np.mean(np.maximum(0.0, 1.0 - preds)) + (lambda_val / 2.0) * np.matmul(w.T, w)
        # if abs(primal_old - primal_new) < opts['tol']:
        #     break
        # primal_old = primal_new

    return w



def global_model(Xtrain, Ytrain, Xtest, Ytest, lambda_val, opts):
    """
    :param Xtrain: List of numpy arrays containing training data. Each numpy array represents a batch of training samples.
    :param Ytrain: List of numpy arrays containing training labels. Each numpy array represents a batch of training labels.
    :param Xtest: List of numpy arrays containing testing data. Each numpy array represents a batch of testing samples.
    :param Ytest: List of numpy arrays containing testing labels. Each numpy array represents a batch of testing labels.
    :param lambda_val: Regularization parameter value.
    :param opts: Dictionary containing additional options.
    :return: The error rate of the global model.

    This method takes in the training and testing data, regularization parameter, and additional options and returns the error rate of the global model. The training data is given as a list
    * of numpy arrays, where each numpy array represents a batch of training samples. The training labels are given as a list of numpy arrays, where each numpy array represents a batch of
    * training labels. The testing data and labels follow the same format.

    The method first concatenates all the training and testing data and labels to obtain a single set of data and labels. Then, it checks the value of the 'obj' key in the 'opts' dictionary
    *. If it is set to 'R', it performs ridge regression to compute the weights 'w'. If it is set to any other value, it calls the 'simple_svm' method to compute the weights 'w'.

    If the 'avg' key in the 'opts' dictionary is True, it computes the error rate for each testing batch separately and then takes the average. Otherwise, it computes the error rate for
    * all testing data together.

    The computed error rate is returned by the method.
    """
    d = Xtrain[0].shape[1]
    allX = np.concatenate(Xtrain)
    allY = np.concatenate(Ytrain)
    allXtest = np.concatenate(Xtest)
    allYtest = np.concatenate(Ytest)
    if opts['obj'] == 'R':
            w = np.linalg.inv(np.transpose(allX) @ allX + lambda_val * np.eye(d)) @ np.transpose(allX) @ allY
            err = np.sqrt(np.mean(np.square(allYtest - allXtest @ w)))
    else:
            w = simple_svm(allX, allY, lambda_val, opts)
            if opts['avg']:
                errs = np.zeros(len(Xtest))
                for t in range(len(Xtest)):
                    predvals = np.sign(Xtest[t] @ w)
                    errs[t] = np.mean(predvals != Ytest[t])
                err = np.mean(errs)
            else:
                predvals = np.sign(allXtest @ w)
                err = np.mean(allYtest != predvals)
    return err

