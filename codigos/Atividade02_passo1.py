import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
# from math import sqrt as R

# x data
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
# t data
t = np.array([[0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839]])
# order polynomial
M = 9
l = [-18, 0]
L = [np.exp(l[0]), np.exp(l[1])]
# order polynomial to code
m = M + 1
for k in range(2):
    # matrix A creation
    A = np.zeros((len(x), m))
    # vector w creation
    w = np.zeros((m, 1))
    # mount matrix A
    xy = np.linspace(0, 1, 100)
    B = np.zeros((len(xy), m))
    # print(xy)
    for i in range(len(x)):
        for j in range(m):
            A[i][j] = x[i] ** j

    for i in range(len(xy)):
        for j in range(m):
            B[i][j] = xy[i] ** j

    if M == 0:
        w[0] = 1/len(t) * np.sum(t)
    else:
        w = inv(A.T.dot(A)+L[k]*np.identity(m)).dot(A.T).dot(t.T)

    print(B)
    print(w)
    y = B.dot(w)

    plt.subplot(1, 2, k+1)
    plt.axis([-0.25, 1.25, -1.25, 1.25])
    plt.scatter(x, t, color='w', edgecolors='b')
    plt.plot(xy, np.sin(2 * np.pi * xy), 'g', alpha=0.5)
    plt.plot(xy, y, 'r', alpha=0.5, label='Order Polynomial: M = {} e ln Lambda = {}'.format(M, l[k]))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('t')
plt.show()
