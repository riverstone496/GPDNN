from kernels import *
from predict import *
from plots import *

import numpy as np
import matplotlib.pyplot as plt

###################
# common setting
# input
x_min = - 5.0
x_max = + 5.0
N = 100
X = np.linspace(x_min, x_max, N)
# noise parameter
sigma2_y = 0.1

###################
###################
# training
N_train_all = 40
X_train_all = np.linspace(-3.0, 3.0, N_train_all)
Y_train_all = np.sin( X_train_all * 2*np.pi / (max(X_train_all) - min(X_train_all)))

n = 40  #訓練データ数
X_train = X_train_all[1:n]
Y_train = Y_train_all[1:n]

#mu_m, sigma2_m = predict(X_train, Y_train, X, kernel_m, sigma2_y)
mu_g, sigma2_g = predict(X_train, Y_train, X, kernel_rbf, sigma2_y)
mu_e, sigma2_e = predict(X_train, Y_train, X, kernel_e, sigma2_y)
mu_r, sigma2_r = predict(X_train, Y_train, X, kernel_r, sigma2_y)
mu_d, sigma2_d = predict(X_train, Y_train, X, kernel_d, sigma2_y)

###################
# plot

plt.figure(figsize=(8,7))

plt.subplot(221)
plot_result(X, X_train, Y_train, mu_g, sigma2_g)
plt.title("RBF")

plt.subplot(222)
plot_result(X, X_train, Y_train, mu_e, sigma2_e)
plt.title("NN(erf)")

plt.subplot(223)
plot_result(X, X_train, Y_train, mu_r, sigma2_r)
plt.title("NN(ReLU)")

plt.subplot(224)
plot_result(X, X_train, Y_train, mu_d, sigma2_d)
plt.title("DNN(ReLU)")

plt.show()

