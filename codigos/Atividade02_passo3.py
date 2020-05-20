import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import sqrt as R


# testing
# x data
xt = np.array([0.03496008, 0.13350487, 0.18029867, 0.24458861, 0.32058624, 0.35986675, 0.48402573, 0.65220266,
              0.93419564, 0.96685168])
# t data
tt = np.array([[0.31494709,  0.79463996,  0.93361892,  1.03720089,  0.96884765, 0.82544549,  0.13033268, -0.73615562,
               -0.34928354, -0.16464626]])
# ==================================================================================================================== #


# x data
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
# t data
t = np.array([[0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839]])
# order polynomial
M = 9
l = [-35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20]
E = []
Et = []
for k in range(len(l)):
    L = np.exp(l[k])
    # order polynomial to code
    m = M + 1
    # matrix A creation
    A = np.zeros((len(x), m))
    At = np.zeros((len(xt), m))
    # vector w creation
    w = np.zeros((m, 1))
    wt = np.zeros((m, 1))

    for i in range(len(x)):
        for j in range(m):
            A[i][j] = x[i] ** j
            At[i][j] = xt[i] ** j

    if M == 0:
        w[0] = 1/len(t) * np.sum(t)
        wt[0] = 1 / len(tt) * np.sum(tt)
    else:
        w = inv(A.T.dot(A)+L*np.identity(m)).dot(A.T).dot(t.T)
        wt = inv(At.T.dot(At) + L * np.identity(m)).dot(At.T).dot(tt.T)

    E.append(R(2 * 0.5 * (A.dot(w) - t.T).T.dot(A.dot(w) - t.T) / 10))
    Et.append(R(2 * 0.5 * (At.dot(wt) - tt.T).T.dot(At.dot(wt) - tt.T) / 10))

# plt.axis([0, 1, -1, 1])
plt.plot(l, E, 'b', label='Treinamento')
plt.plot(l, Et, 'r', label='Teste')
plt.title('Erms')
plt.legend()
plt.show()
