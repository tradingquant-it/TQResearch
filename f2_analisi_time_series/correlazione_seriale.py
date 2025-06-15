import numpy as np
import matplotlib.pyplot as plt

# Impostare il seed per la riproducibilità
np.random.seed(1)
x = np.arange(1, 101) + 20.0 * np.random.normal(size=100)

np.random.seed(2)
y = np.arange(1, 101) + 20.0 * np.random.normal(size=100)

# Creare il grafico
plt.scatter(x, y)
plt.show()

sigma = np.cov(x,y)[0,1]
print(sigma)

corr = np.corrcoef(x,y)[0,1]
print(corr)