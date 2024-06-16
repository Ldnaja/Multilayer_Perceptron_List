# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sympy as sp

# Class points
w1 = np.array([[0.1, -0.2], [0.2, 0.1], [-0.15, 0.2], [1.1, 0.8], [1.2, 1.1]])
w2 = np.array([[1.1, -0.1], [1.25, 0.15], [0.9, 0.1], [0.1, 1.2], [0.2, 0.9]])

# Visualization of points
plt.scatter(w1[:, 0], w1[:, 1], color='blue', label='Classes w1')
plt.scatter(w2[:, 0], w2[:, 1], color='red', label='Classes w2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Points from Classes w1 and w2')
plt.show()

# Linear Separability Check
X = np.vstack((w1, w2))
y = np.array([0]*len(w1) + [1]*len(w2))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=1000)
clf.fit(X_scaled, y)

# Visualization of the decision boundary
xx, yy = np.meshgrid(np.linspace(-1.5, 2, 500), np.linspace(-1.5, 2, 500))
Z = clf.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multilayer Perceptron Decision Frontier')
plt.show()

# Lines
x = np.linspace(-1, 1, 400)
y1 = -x
y2 = x
y3 = np.full_like(x, 0.25)

plt.plot(x, y1, label='x + y = 0')
plt.plot(x, y2, label='x - y = 0')
plt.plot(y3, x, label='x = 1/4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Lines in 2D Space')
plt.grid(True)
plt.show()

clf = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=5000)
clf.fit(patterns, labels)

print("Are the patterns linearly separable?", clf.score(patterns, labels) == 1)

n = sp.symbols('n')
tansig = (sp.exp(n) - sp.exp(-n)) / (sp.exp(n) + sp.exp(-n))
tansig_prime = sp.diff(tansig, n)
tansig_prime_simplified = sp.simplify(tansig_prime)

print("Derivative of the tansig function:", tansig_prime_simplified)
