import  numpy as np
from kernels import *

def calc_kernel_sequence(X, x, kernel):
    N = len(X)
    seq = [kernel(X[n], x) for n in range(N)]
    return np.array(seq)

def predict(X_train, Y_train, X, kernel, sigma2_y):
    N = len(X)
    N_train = len(Y_train)

    mu = np.zeros(N)
    sigma2 = np.zeros(N)

    K = calc_kernel_matrix(X_train, kernel)
    invmat = np.linalg.inv(sigma2_y * np.eye(N_train) + K)

    for n in range(N):
        seq = calc_kernel_sequence(X_train, X[n], kernel) # N dim
        mu[n] = seq.T @ invmat @ Y_train
        sigma2[n] = sigma2_y + kernel(X[n], X[n]) - seq.T@invmat@seq

    return mu, sigma2
