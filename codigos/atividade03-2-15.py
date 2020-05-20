# author : Natália Freitas Araújo
# subject : Machine Learning
# UFPA - PPCA - NDAE
# Activity 3.2 - 1.5
import matplotlib.pyplot as plt
import pandas as pd
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
M = np.array([0, 1, 6, 9])
# order polynomial to code
m = np.array(M + 1)
W = np.zeros((10, 4))
# ==================================================================================================================== #
for k in range(4):
    w = np.zeros((m[k], 1))
    A = np.zeros((len(x), m[k]))
    mi = np.linspace(x[0], x[-1], m[k])
    s = np.var(t)
    Phi = lambda X, MI, S: 1/(1+np.exp(-(X-MI)/S))
    # matrix A calculation => Phi @ x ** j
    for i in range(len(x)):
        for j in range(m[k]):
            A[i][j] = Phi(X=x[i], MI=mi[j], S=s)
    # vector w calculation => w = (A'*A)**(-1)*(A'*t)
    if M[k] == 0:
        w[0] = 1 / len(t) * np.sum(t)
    else:
        w = inv(A.T.dot(A)).dot(A.T).dot(t.T)
    # ----------------------------------------- #
    for i in range(len(w)):
        W[i][k] = w[i]
# ==================================================================================================================== #
print('\n\n')
print(' '*15, 'Tabela Atividade 3.2\n')
W = pd.DataFrame(W, columns=['M = 0', 'M = 1', 'M = 6', 'M = 9'])
print(W)
