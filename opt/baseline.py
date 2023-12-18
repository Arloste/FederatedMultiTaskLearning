import numpy as np

def local_model(Xtrain, Ytrain, Xtest, Ytest, lambda_val, opts):
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

import numpy as np


def simple_svm_v2(X, y, lambda_val, opts):
    n, d = X.shape
    w = np.zeros((d, 1))
    alpha = np.zeros((n, 1))
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
            grad = lambda_val * n * (1.0 - update) / (np.matmul(curr_x, np.transpose(curr_x)) + (alpha_old * curr_y))
            # apply update
            alpha[i] = curr_y * np.maximum(0, np.minimum(1.0, grad))  # removed an unnecessary 'alpha[i] ='
            w = w + ((alpha[i] - alpha_old) * np.transpose(curr_x) * (1.0 / (lambda_val * n)))

        # break if less than tol
        preds = y * np.matmul(X, w)
        primal_new = np.mean(np.maximum(0.0, 1.0 - preds)) + (lambda_val / 2.0) * np.matmul(np.transpose(w), w)
        if abs(primal_old - primal_new) < opts['tol']:
            break
        primal_old = primal_new

    return w
def simple_svm(X, y, lambda_val, opts):
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

