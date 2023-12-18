from scipy.linalg import sqrtm

from opt.mocha import compute_primal, compute_rmse
import numpy as np
import scipy


def compute_min(diagonal):
    diagonal = [i if i > 1e-7 else 1e-7 for i in diagonal]
    diagonal = np.array(diagonal)
    return diagonal




def run_mbsgd(Xtrain, Ytrain, Xtest, Ytest, lambda_, opts, problem_type='R', avg=True):

    m = len(Xtrain)  # number of tasks
    d = Xtrain[0].shape[1]  # number of features
    W = np.random.randn(d, m)
    Sigma = np.eye(m) / m
    Omega = np.linalg.inv(Sigma)
    totaln = 0
    n = np.zeros(m)
    for t in range(m):
        n[t] = len(Ytrain[t])
        totaln += n[t]

    if opts["update"]:
        rmse = np.zeros(opts["mbsgd_inner_iters"])

        primal_objs = np.zeros(opts["mbsgd_inner_iters"])
    else:
        rmse = np.zeros(opts["mbsgd_outer_iters"])

        primal_objs = np.zeros(opts["mbsgd_outer_iters"])

    for h in range(opts["mbsgd_outer_iters"]):
        # note h if used then +1
        for hh in range(opts["mbsgd_inner_iters"]):
            np.random.seed((h + 1) * 1000)
            if opts["sys_het"]:
                sys_iters = (opts['top'] - opts['bottom']) * np.random.random_sample((m,1)) + opts['bottom']

            rmse[hh] = compute_rmse(Xtest, Ytest, W,type= problem_type)
            primal_objs[hh] = compute_primal(Xtrain, Ytrain, W, Omega, lambda_)

            total_loss = np.zeros((d, m))
            local_iters = np.zeros(m)

            for t in range(m):
                tperm = np.random.permutation(int(n[t]))
                if opts["sys_het"]:
                    pass

                else:
                    local_iters[t] = n[t] * opts["mbsgd_sgd_frac"]

                # get local gradients
                for s in range(int(local_iters[t])):
                    idx = tperm[int(np.mod((s + 1), n[t]))]
                    curr_y = Ytrain[t][idx]
                    curr_x = Xtrain[t][idx, :]

                    if np.dot(curr_y * curr_x, W[:, t]) < 1.0:
                        update = curr_y * curr_x # equivalent to curr_y .* curr_x'
                        total_loss[:, t] = total_loss[:, t] + update

                denom = sum(local_iters[:])

                W = W @ (np.eye(m) - (Omega * (opts['mbsgd_scaling'] / (hh + 1)))) + (
                            opts['mbsgd_scaling'] / denom * total_loss)



        A = np.dot(W.T, W)
        if any(np.linalg.eigvals(A) < 0):
            V, D = np.linalg.eig(A)
            D[D <= 1e-7] = 1e-7
            A = np.dot(V * D, V.T)

        sqm = sqrtm(A)
        Sigma = sqm / np.trace(sqm)
        Omega = np.linalg.inv(Sigma)
    return rmse, primal_objs, W



