# author : Natália Freitas Araújo
# subject : Machine Learning
# UFPA - PPCA - NDAE
# Activity 1.3
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
# data input x
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
# x = np.array(np.random.rand(10))
# target => sin(2 * pi * x) + noise
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])
# t = np.array(np.sin(2*np.pi*x)+0.5*np.random.rand(10))
# data test
x_test = np.linspace(0, 1, 100)
# ==================================================================================================================== #
# order polynomial
M = np.array([0, 1, 3, 9])
# order polynomial to code
m = np.array(M + 1)
for k in range(4):
    w = np.zeros((m[k], 1))
    A = np.zeros((len(x), m[k]))
    # matrix A calculation => Phi @ x ** j
    for i in range(len(x)):
        for j in range(m[k]):
            A[i][j] = x[i] ** j

    if M[k] == 0:
        w[0] = 1 / len(t) * np.sum(t)
    else:
        w = inv(A.T.dot(A)).dot(A.T).dot(t.T)

    B = np.zeros((len(x_test), m[k]))
    for i in range(len(x_test)):
        for j in range(m[k]):
            B[i][j] = x_test[i] ** j

    y = B.dot(w)

    plt.subplot(2, 2, k+1)
    plt.axis([0, 1, -2, 2])
    plt.scatter(x, t, color='w', edgecolors='b')
    plt.plot(x_test, np.sin(2 * np.pi * x_test), 'g', alpha=0.5)
    plt.plot(x_test, y, 'r', alpha=0.5, label='M = {}'.format(M[k]))
    plt.legend()
    # plt.title(f'M = {M[k]} ')
    plt.xlabel('x')
    plt.ylabel('t')
    print(f'M{M[k]}')
    print(w)

plt.show()
