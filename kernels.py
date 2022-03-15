import numpy as np

def calc_kernel_matrix(X, kernel):
    N = len(X)
    K = np.zeros((N,N))
    for n1 in range(N):
        for n2 in range(n1,N):
            K[n1, n2] = kernel(X[n1], X[n2])
            K[n2, n1] = K[n1,n2]
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

# neural network (erf) covariance function
def aug(x) :
        return np.array([1, x])

Sigma = 10.0*np.eye(2)
def kernel_e(x1, x2) :
    return (2/np.pi) * np.arcsin((2*np.trace(aug(x2)@aug(x1).T*Sigma))/np.sqrt((1+2*np.trace(aug(x1)@aug(x1).T*Sigma))*(1+2*np.trace(aug(x2)@aug(x2).T*Sigma))))
def K_e(X): 
    return calc_kernel_matrix(X, kernel_e)

# neural network (ReLU) covariance function
def rad(x1, x2):
    return np.arccos(max(min(sum(x1.T*x2)/(np.linalg.norm(x1)*np.linalg.norm(x2)), 1.0), -1.0))
def kernel_r(x1, x2) :
    return (1.0 / np.pi) * np.linalg.norm(aug(x1)) * np.linalg.norm(aug(x2)) * (np.sin(rad(aug(x1), aug(x2))) + (np.pi - rad(aug(x1), aug(x2)))*np.cos(rad(aug(x1), aug(x2))))
def K_r(X):
    return calc_kernel_matrix(X, kernel_r)

# deep neural network (ReLU) covariance function

#層の深さを指定
L = 8

def kernel_tmp(x1, x2, L):
    sigma2_b = 1.0
    sigma2_w = 2.0

    if( L > 0 ):
        k_11=kernel_tmp(x1, x1, L-1)
        k_22=kernel_tmp(x2, x2, L-1)
        theta=np.arccos(kernel_tmp(x1, x2, L-1) / np.sqrt(k_11 * k_22))
        return sigma2_b + (sigma2_w/(2*np.pi))*np.sqrt(k_11*k_22)*(np.sin(theta) + (np.pi - theta)*np.cos(theta)) 
    else:
        return sigma2_b + sigma2_w*(x1.T*x2)

def kernel_d(x1, x2):
    return kernel_tmp(x1, x2, L)

def K_d(X):
    return calc_kernel_matrix(X, kernel_d)