
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# 1. Generazione del random walk zₜ
z = np.zeros(1000)
for i in range(1, 1000):
    z[i] = z[i-1] + np.random.normal()


# Plot di zₜ
plt.plot(z)
plt.show()


from statsmodels.graphics.tsaplots import plot_acf

# 2. Correlogramma di zₜ e delle sue differenze
plot_acf(z, title="Serie z", alpha=0.05, lags=30)
plot_acf(np.diff(z), title="Variazioni assolute della Serie z", alpha=0.05, lags=30)
plt.show()



x = 0.3 * z + np.random.normal(size=1000)
y = 0.6 * z + np.random.normal(size=1000)


plt.plot(x)
plt.title("Serie x")
plt.show()
plt.plot(y)
plt.title("Serie y")
plt.show()

comb = 2 * x - y

plt.plot(comb)
plt.title("Serie 2x - y")

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(comb, lags=30)
plt.show()

from arch.unitroot import ADF

adf = ADF(comb)
print(adf.summary())

from arch.unitroot import PhillipsPerron

php = PhillipsPerron(comb)
print(php.summary())


from arch.unitroot.cointegration import phillips_ouliaris

po = phillips_ouliaris(2*x, -1*y)
print(po.summary())

