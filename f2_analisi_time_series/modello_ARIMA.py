
#############################################################
#                   ARIMA(1,1,1)
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
matplotlib.use('TkAgg')

# Imposta il seed per la riproducibilità
np.random.seed(1)

# Specifica i parametri ARIMA(1,1,1)
ar = np.array([1, -0.6])   # AR(1) con coefficiente 0.6
ma = np.array([1, -0.5])   # MA(1) con coefficiente -0.5

# Simula un processo ARMA(1,1)
arma_process = ArmaProcess(ar, ma)
x_diff = arma_process.generate_sample(nsample=1000)

# Integra la serie per ottenere ARIMA(1,1,1)
x = np.cumsum(x_diff)

# Plot della serie simulata
plt.plot(x)
plt.show()


from statsmodels.tsa.arima.model import ARIMA

# Stima del modello ARIMA(1,1,1)
model = ARIMA(x, order=(1, 1, 1))
x_arima = model.fit()

# Sommario dei risultati
print(x_arima.summary())

# Intervalli di confidenza al 95%
ar1 = x_arima.params[0]
ma1 = x_arima.params[1]

se_ar1 = x_arima.bse[0]
se_ma1 = x_arima.bse[1]

ci_ar1 = ar1 + np.array([-1.96, 1.96]) * se_ar1
ci_ma1 = ma1 + np.array([-1.96, 1.96]) * se_ma1

print("IC 95% AR(1):", ci_ar1)
print("IC 95% MA(1):", ci_ma1)

# Correlogramma dei residui
from statsmodels.graphics.tsaplots import plot_acf

residuals = x_arima.resid
plot_acf(residuals,  lags=30, alpha=0.05)
plt.show()


# Test di Ljung-Box sui residui
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box = acorr_ljungbox(residuals, lags=[20], return_df=True)
print(ljung_box)

################################################
#         AMZN
#

import numpy as np
import yfinance as yf
import pandas as pd


symbol = 'AMZN'
data = yf.download(symbol, start="2013-01-01", end="2015-09-03", group_by='ticker', auto_adjust=False)
amzn_data = data[symbol]

amzn_log_returns = np.log(amzn_data['Close']).diff().dropna()
amzn_log_returns.index = pd.DatetimeIndex(amzn_log_returns.index).to_period("D")


from statsmodels.tsa.arima.model import ARIMA

best_aic = np.inf
best_order = None
best_model = None

for p in range(1, 5):
    for d in range(0, 2):
        for q in range(1, 5):
            try:
                model = ARIMA(amzn_log_returns, order=(p, d, q))
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

# Test di Ljung-Box sui residui
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box = acorr_ljungbox(residuals, lags=[20], return_df=True)
print(ljung_box)

# === Forecast ===
# Ottieni l'ultima data e converti da Period a Timestamp
last_date = amzn_log_returns.index[-1].to_timestamp()
# Crea intervallo con frequenza giornaliera
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=25, freq='D')

# Previsione
forecast_result = best_model.get_forecast(steps=25)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

# Assegna indice alle previsioni
forecast_mean.index = forecast_dates
conf_int.index = forecast_dates

plt.figure(figsize=(10, 5))
plt.plot(amzn_log_returns.index.to_timestamp(), amzn_log_returns, label='Serie Storica')
plt.plot(forecast_mean.index, forecast_mean, label='Previsione')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='blue', alpha=0.2, label='Intervallo 95%')
plt.legend()
plt.title("Previsione a 25 giorni dei ritorni logaritmici di AMZN")
plt.show()


################################################
#         SP500
#
import numpy as np
import yfinance as yf
import pandas as pd


symbol = '^GSPC'
data = yf.download(symbol, start="2013-01-01", end="2015-09-03", group_by='ticker', auto_adjust=False)
sp500_data = data[symbol]

sp500_log_returns = np.log(sp500_data['Close']).diff().dropna()
sp500_log_returns.index = pd.DatetimeIndex(sp500_log_returns.index).to_period("D")


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

# Test di Ljung-Box sui residui
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box = acorr_ljungbox(residuals, lags=[20], return_df=True)
print(ljung_box)

# === Forecast ===
# Crea intervallo con frequenza giornaliera
last_date = sp500_log_returns.index[-1].to_timestamp()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=25, freq='D')

# Previsione
forecast_result = best_model.get_forecast(steps=25)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

# Assegna indice alle previsioni
forecast_mean.index = forecast_dates
conf_int.index = forecast_dates

plt.figure(figsize=(10, 5))
plt.plot(sp500_log_returns.index.to_timestamp(), sp500_log_returns, label='Serie Storica')
plt.plot(forecast_mean.index, forecast_mean, label='Previsione')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='blue', alpha=0.2, label='Intervallo 95%')
plt.legend()
plt.title("Previsione a 25 giorni dei ritorni logaritmici di sp500")
plt.show()
print("")