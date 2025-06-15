
#############################################################
#                   GARCH(1,1).
#
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(2)

a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(0, 1, n)
eps = np.zeros(n)
sigsq = np.zeros(n)

for i in range(1, n):
    sigsq[i] = a0 + a1 * (eps[i-1]**2) + b1 * sigsq[i-1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

plot_acf(eps, lags=40)
plt.show()

plot_acf(eps**2, lags=40)
plt.show()

from arch import arch_model

# Stimiamo un modello GARCH(1,1)
model = arch_model(eps, vol='GARCH', p=1, q=1)
res = model.fit(disp="off")

# Stampiamo l'intervallo di confidenza al 97.5%
print(res.summary())
conf_int = res.conf_int(alpha=0.05)  # livello di confidenza 95%
print("Intervalli di confidenza al 97.5%:")
print(conf_int)



#############################################################
#                   S&P500
#

import numpy as np
import yfinance as yf
import pandas as pd

symbol = '^GSPC'
data = yf.download(symbol, start="2007-01-01", end="2015-10-01", group_by='ticker', auto_adjust=False)
sp500_data = data[symbol]

sp500_log_returns = np.log(sp500_data['Adj Close']).diff().dropna()
sp500_log_returns.index = pd.DatetimeIndex(sp500_log_returns.index).to_period("D")

import matplotlib.pyplot as plt
# sp500_log_returns.plot()
# plt.show()

from statsmodels.tsa.arima.model import ARIMA

best_aic = np.inf
best_order = None
best_model = None

for p in range(1, 5):
    for d in range(0, 2):
        for q in range(1, 5):
            try:
                model = ARIMA(sp500_log_returns, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
                    best_model = result
            except:
                continue

print("Ordine ARIMA ottimale:", best_order)


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

residuals = best_model.resid
plot_acf(residuals, lags=30)
plt.show()

squared_residuals = residuals**2
plot_acf(squared_residuals, lags=30)
plt.show()

from arch import arch_model

# Stimiamo un modello GARCH(1,1)
garch_model = arch_model(sp500_log_returns, vol='GARCH', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Stampiamo l'intervallo di confidenza al 97.5%
print(garch_fit.summary())

# Rimuove il primo valore NA dai residui
res = garch_fit.resid.dropna()

# ACF dei residui GARCH
plot_acf(res, lags=30)
plt.show()

# ACF dei residui quadratici GARCH
plot_acf(res**2, lags=30)
plt.show()
