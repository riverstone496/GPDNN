import matplotlib.pyplot as plt
import numpy as np

x_min = -5
x_max = 5
y_max = 5
y_min = -2

def plot_result(X, X_train, Y_train, mu, sigma2):
    plt.plot(X_train, Y_train, "xk")
    plt.fill_between(X, mu + np.sqrt(sigma2), mu - np.sqrt(sigma2), color="c", alpha=0.5)
    plt.plot(X, mu,"-b")
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])