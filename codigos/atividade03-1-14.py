# author : Natália Freitas Araújo
# subject : Machine Learning
# UFPA - PPCA - NDAE
# Activity 3.1 -  1.4
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
# data input x
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
# target => sin(2 * pi * x) + noise
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])
# data test
N = 100
x_test = np.linspace(0, 1, N)
t_test = np.array(np.sin(x_test)+np.random.rand(N))
# ==================================================================================================================== #
# order polynomial
M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# order polynomial to code
m = np.array(M)
E = []
Et = []
for k in range(10):
    mi = np.linspace(x[0], x[-1], m[k])
    s = np.var(t)
    Phi = lambda X, MI, S: np.exp(-((X - MI) ** 2) / 2 * S ** 2)
    w = np.zeros((m[k], 1))
    A = np.zeros((len(x), m[k]))
    # matrix A calculation => Phi @ x ** j
    for i in range(len(x)):
        for j in range(m[k]):
            A[i][j] = Phi(X=x[i], MI=mi[j], S=s)

    B = np.zeros((len(x_test), m[k]))
    for i in range(len(x_test)):
        for j in range(m[k]):
            B[i][j] = Phi(X=x_test[i], MI=mi[j], S=s)

    if M == 0:
        w[0] = 1 / len(t) * np.sum(t)
    else:
        w = inv(A.T.dot(A)).dot(A.T).dot(t.T)

    E.append(np.sqrt(2 * 0.5 * (A.dot(w) - t.T).T.dot(A.dot(w) - t.T) / 10))
    Et.append(np.sqrt(2 * 0.5 * (B.dot(w) - t_test.T).T.dot(B.dot(w) - t_test.T) / N))

plt.axis([-0.5, 10, -0.5, 5])
plt.scatter(M, E, color='w', edgecolors='b')
plt.plot(M, E, 'b', label='Treinamento')
plt.scatter(M, Et, color='w', edgecolors='r')
plt.plot(M, Et, 'r', label='Teste')
plt.title('Erms')
plt.legend()
plt.show()

