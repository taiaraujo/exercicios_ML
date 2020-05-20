from sklearn import datasets
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Classes C2 and C3

iris = datasets.load_iris()
X1 = np.array(iris.data[50:100, :])
X2 = np.array(iris.data[100:150, :])
X = np.concatenate((X1, X2))

m1 = np.array([0.02*np.sum(X1[:, n]) for n in range(4)])
m2 = np.array([0.02*np.sum(X2[:, n]) for n in range(4)])

print('m1 = ', m1)
print('m2 = ', m2)

sw = X1.T.dot(X1)-m1.T.dot(m1) + X2.T.dot(X2)-m2.T.dot(m2)

print('Sw = ', sw)

w = inv(sw).dot(m2-m1)

print('w = ', w)

y = X.dot(w)

plt.figure()
colors = ['turquoise', 'orange']
lw = 2

y = np.where(y < 0, 0, 1)

# print(X[y == 0, 0])
# print(X[y == 1, 0])
for color, i in zip(colors, [0, 1]):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA IRIS C2 X C3')
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
plt.show()
