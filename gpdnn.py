from kernels import *
from predict import *
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

mu_g, sigma2_g = predict(X_train, Y_train, X, kernel_rbf, sigma2_y)
#mu_g, sigma2_g = predict(X_train, Y_train, X, kernel_m, sigma2_y)

###################
# plot
y_max = +3
y_min = -3

fig, ax = plt.subplots()

ax.plot(X_train, Y_train, "xk")
ax.fill_between(X, mu_g + np.sqrt(sigma2_g), mu_g - np.sqrt(sigma2_g), color="c", alpha=0.5)
ax.plot(X, mu_g,"-b")

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
plt.show()

print(sigma2_g)