import numpy as np
import scipy


def compute_primal(X, Y, W, Omega, lambda_):
    total_loss = 0
    for t in range(len(X)):
        preds = Y[t] @ (X[t] @ W[:, t])
        total_loss += np.maximum(0.0, 1.0 - preds).mean()
    primal_obj = total_loss + lambda_ / 2 * np.trace((W @ Omega) @ W.T)
    return primal_obj


def compute_dual(alpha, Y, W, Omega, lambda_):
    total_alpha = 0
    for tt in range(len(Y)):
        total_alpha += (-1.0 * alpha[tt] * Y[tt]).mean()
    dual_obj = -lambda_ / 2 * np.trace(np.dot(np.dot(W, Omega), W.T)) - total_alpha
    return dual_obj


def compute_rmse(X, Y, W, type='R', avg=True):
    m = len(X)
    Y_hat = [0] * m
    for t in range(m):
        if type == 'R':
            Y_hat[t] = np.dot(X[t], W[:, t])
        else:
            Y_hat[t] = np.sign(np.dot(X[t], W[:, t]))

    if avg:
        all_errs = [0] * m
        for t in range(m):
            if type == 'R':
                all_errs[t] = np.sqrt(((Y[t] - Y_hat[t]) ** 2).mean())
            else:
                all_errs[t] = (Y[t] != Y_hat[t]).mean()
        err = np.mean(all_errs)
    else:
        Y = np.ravel(Y)
        Y_hat = np.ravel(Y_hat)
        if type == 'R':
            err = np.sqrt(((Y - Y_hat) ** 2).mean())
        else:
            err = (Y != Y_hat).mean()
    return err


def run_mocha(Xtrain, Ytrain, Xtest, Ytest, lambda_, opts, problem_type='R', avg=True):

    m = len(Xtrain)  # number of tasks
    d = Xtrain[0].shape[1]  # number of features
    W = np.random.randn(d, m)
    alpha = [np.zeros(len(y)) for y in Ytrain]
    Sigma = np.eye(m) / m
    Omega = np.linalg.inv(Sigma)
    totaln = 0
    n = np.zeros(m)
    for t in range(m):
        n[t] = len(Ytrain[t])
        totaln += n[t]

    if opts["update"]:
        rmse = np.zeros(opts["mocha_inner_iters"])
        dual_objs = np.zeros(opts["mocha_inner_iters"])
        primal_objs = np.zeros(opts["mocha_inner_iters"])
    else:
        rmse = np.zeros(opts["mocha_outer_iters"])
        dual_objs = np.zeros(opts["mocha_outer_iters"])
        primal_objs = np.zeros(opts["mocha_outer_iters"])

    for h in range(opts["mocha_outer_iters"]):
        if not opts["update"]:
            curr_err = compute_rmse(Xtest, Ytest, W)
            rmse[h] = curr_err
            primal_objs[h] = compute_primal(Xtrain, Ytrain, W, Omega, lambda_)
            dual_objs[h] = compute_dual(alpha, Ytrain, W, Omega, lambda_)


        deltaW = np.zeros((d, m))
        deltaB = np.zeros((d, m))
        for t in range(m):
            np.random.shuffle(Ytrain[t])
            alpha_t = alpha[t].copy()
            curr_sig = Sigma[t, t]
            if opts["sys_het"]:
                local_iters = int(n[t] * (opts["top"] - opts["bottom"]) * np.random.rand() + opts["bottom"])
            else:
                local_iters = int(n[t] * opts["mocha_sdca_frac"])

            for s in range(local_iters):
                idx = int(s % n[t])
                alpha_old = alpha_t[idx]
                curr_y = Ytrain[t][idx]
                curr_x = Xtrain[t][idx, :]

                update = curr_y * np.dot(curr_x, (W[:, t] + deltaW[:, t]))

                grad = lambda_ * n[t] * (1.0 - update) / (curr_sig * (np.dot(curr_x, curr_x)) + alpha_old * curr_y)

                alpha_t[idx] = curr_y * max(0.0, min(1.0, grad))

                deltaW[:, t] += Sigma[t, t] * (alpha_t[idx] - alpha_old) * curr_x / (lambda_ * n[t])
                deltaB[:, t] += (alpha_t[idx] - alpha_old) * curr_x / n[t]
                alpha[t] = alpha_t

        for t in range(m):
            for tt in range(m):
                W[:, t] += deltaB[:, tt] * Sigma[t, tt] / lambda_

        A = np.dot(W.T, W)
        if any(np.linalg.eigvals(A) < 0):
            V, D = np.linalg.eig(A)
            D[D <= 1e-7] = 1e-7
            A = np.dot(V * D, V.T)

        sqm = scipy.linalg.sqrtm(A)
        Sigma = sqm / np.trace(sqm)
        Omega = np.linalg.inv(Sigma)
    return rmse, primal_objs, dual_objs,W



