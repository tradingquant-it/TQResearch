from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
s = np.random.normal(size=100)

acf_coef = acf(s)
plot_acf(acf_coef, lags=20)

plt.show()

w = np.arange(1, 100)
acf_coef = acf(w, nlags=100)
plot_acf(acf_coef, lags=20)
plt.show()

a = np.arange(1,11)
s = []
for i in range (0,10):
    for j in range (0,10):
        s.append(a[j])

acf_coef = acf(s)
plot_acf(acf_coef, lags=20)
plt.show()