import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

np.random.seed(1)
whiteNoise = np.random.standard_normal(1000)

plt.figure(figsize=[10, 7.5]); # Dimensioni del grafico
plt.plot(whiteNoise)
plt.title("Simulated White Noise")
plt.show()

acf_coef = acf(whiteNoise)
plot_acf(acf_coef, lags=30)
plt.show()

var = np.var(whiteNoise)
print(var)

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
steps = np.random.standard_normal(1000)
steps[0] = 0
random_walk = np.cumsum(steps)

plt.figure(figsize=[10, 7.5]); # Dimensioni del grafico
plt.plot(random_walk)
plt.title("Simulated Random Walk")
plt.show()

random_walk_acf_coef = acf(random_walk)
plot_acf(random_walk, lags=20)
plt.show()


import datetime
import yfinance as yf

symbol = 'MSFT'
start_date = datetime.datetime(2000, 1, 1)
end_date = datetime.datetime(2017, 1, 8)
mstf = yf.download(
        symbol,
        start=start_date - datetime.timedelta(days=365),
        end=end_date,
        group_by='ticker', auto_adjust=False
    )
mstf = mstf[symbol]

mstf_close = mstf['Adj Close']
msft_diff = np.diff(mstf_close, n=1)

msft_diff_acf_coef = acf(msft_diff,missing="drop")
plot_acf(msft_diff_acf_coef, lags=20)
plt.show()


symbol = '^GSPC'
sp500 = yf.download(
        symbol,
        start=start_date - datetime.timedelta(days=365),
        end=end_date,
        group_by='ticker', auto_adjust=False
    )
sp500 = sp500[symbol]
sp500_close = sp500['Adj Close']

sp500_diff = np.diff(sp500_close, n=1)
sp500_diff_acf_coef = acf(sp500_diff,missing="drop")
plot_acf(sp500_diff_acf_coef, lags=20)
plt.show()