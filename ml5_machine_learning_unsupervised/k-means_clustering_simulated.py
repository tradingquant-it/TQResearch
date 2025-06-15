# simulated_data.py

import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(1)

# Imposto il numero di campioni, la media e la
# varianza per ognuno dei tre cluster simulati
samples = 100
mu = [(7, 5), (8, 12), (1, 10)]
cov = [
    [[0.5, 0], [0, 1.0]],
    [[2.0, 0], [0, 3.5]],
    [[3, 0], [0, 5]],
]

# Generazione di una lista di cluster 2D
norm_dists = [
    np.random.multivariate_normal(m, c, samples)
    for m, c in zip(mu, cov)
]
X = np.array(list(itertools.chain(*norm_dists)))

# Applico l'algoritmo K-Means per k=3, che è uguale
# al numero dei veri cluster gaussiani
km3 = KMeans(n_clusters=3)
km3.fit(X)
km3_labels = km3.labels_

# Applico l'algoritmo K-Means per k=4, che è uguale
# al numero dei veri cluster gaussiani
km4 = KMeans(n_clusters=4)
km4.fit(X)
km4_labels = km4.labels_

# Creo il grafico per confrontare l'algoritmo
# K-Means per k=3 e k=4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
ax1.scatter(X[:, 0], X[:, 1], c=km3_labels.astype(float))
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_title("K-Means con $k=3$")
ax2.scatter(X[:, 0], X[:, 1], c=km4_labels.astype(float))
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_title("K-Means con $k=4$")
plt.show()
print("")
