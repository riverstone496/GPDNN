import numpy as np

def calc_kernel_matrix(X, kernel):
    N = len(X)
    K = np.zeros((N,N))
    for n1 in range(N):
        for n2 in range(N):
            K[n1, n2] = kernel(X[n1], X[n2])
    return np.array(K)

# multinomial covariance function
M = 3

def multi(x, M) :
    return  np.array([x**m for m in range(M+1)])

def kernel_m(x1, x2) :
    Sigma_W = np.diag([10.0, 1.0, 0.1, 0.01])
    return np.trace(multi(x1, M)@multi(x2, M).T*Sigma_W)

def K_mul(X):
    return calc_kernel_matrix(X, kernel_m)

# RBF covariance function
def kernel_rbf(x1, x2):
    alpha = 1.0
    beta = 1.0
    return alpha * np.exp( (-0.5*(x1 - x2)**2) / (beta**2) )

def K_rbf(X):
    return calc_kernel_matrix(X,kernel_rbf)

