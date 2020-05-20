# author : Natália Freitas Araújo
# subject : Machine Learning
# UFPA - PPCA - NDAE
# Activity 2.1
import pandas as pd
import numpy as np
from numpy.linalg import inv

# x data
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
# t data
t = np.array([[0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839]])
# order polynomial
M = 9
l = [-np.infty, -35, 0]
L = [np.exp(l[0]), np.exp(l[1]), np.exp(l[2])]
# order polynomial to code
m = M + 1
W = np.zeros((10, 3))
for k in range(3):
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
    for i in range(len(w)):
        W[i][k] = w[i]
print('\n\n')
W = pd.DataFrame(W, columns=['ln lambda = -inf', 'ln lambda = -18', 'ln lambda = 0'])
print(W)

