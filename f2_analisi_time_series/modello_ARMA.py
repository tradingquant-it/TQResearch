"""
#############################################################
#                   ARMA(1,1)
#

Possiamo iniziare con il più semplice modello ARMA non banale: il modello ARMA(1,1).
Si tratta di un modello autoregressivo di ordine uno combinato con un modello a media mobile di ordine uno.
Un modello di questo tipo ha solo due coefficienti, φ e θ, che rappresentano rispettivamente il primo ritardo
della serie temporale e il termine di rumore bianco ritardato ("shock").

Il modello è definito dalla seguente equazione:
xₜ = φ·xₜ₋₁ + wₜ + θ·wₜ₋₁
Dobbiamo specificare i coefficienti prima della simulazione. Supponiamo:
φ = 0.5
θ = -0.5

In Python, possiamo scrivere:
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess

# Imposta il seed per la riproducibilità
np.random.seed(1)

# Definizione dei parametri ARMA(1,1)
phi = 0.5   # coefficiente AR
theta = -0.5  # coefficiente MA

# Definizione del modello (attenzione al segno)
ar = np.array([1, -phi])     # coefficiente AR in forma caratteristica
ma = np.array([1, theta])    # coefficiente MA

# Simulazione
np.random.seed(1)
arma_process = ArmaProcess(ar, ma)
x = arma_process.generate_sample(nsample=1000)

# Plot della serie simulata
plt.plot(x)
plot_acf(x, lags=30)
plt.show()


# Dal correlogramma possiamo vedere che non c'è un'autocorrelazione significativa, come prevedibile da un modello ARMA(1,1).

# Infine, proviamo a determinare i coefficienti e i loro errori standard utilizzando la funzione ARMA della libreria
# Statsmodels:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(x, order=(1, 0, 1))
result = model.fit()
print(result.summary())


# Possiamo calcolare gli intervalli di confidenza al 95% per ciascun parametro utilizzando gli errori standard:


ar1 = result.params[1]
ma1 = result.params[2]

se_ar1 = result.bse[1]
se_ma1 = result.bse[2]

# Intervalli di confidenza 95%
ci_ar1 = ar1 + np.array([-1.96, 1.96]) * se_ar1
ci_ma1 = ma1 + np.array([-1.96, 1.96]) * se_ma1

print("IC 95% AR(1):", ci_ar1)
print("IC 95% MA(1):", ci_ma1)


# Gli intervalli di confidenza contengono i valori veri dei parametri in entrambi i casi, ma è importante notare
# che sono piuttosto ampi, a causa degli errori standard relativamente grandi.


#############################################################
#                   ARMA(2,2)  negativo
#

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(1)

# Parametri del modello
ar = np.array([1, -0.5, 0.25])   # nota: il segno è invertito per φ₂
ma = np.array([1, 0.5, -0.3])

# Simulazione della serie
np.random.seed(1)
arma_process = ArmaProcess(ar, ma)
x = arma_process.generate_sample(nsample=1000)

# Grafico della serie simulata
plt.plot(x)
plot_acf(x)
plt.show()
print("")

from statsmodels.tsa.arima.model import ARIMA

# Stima del modello ARMA(2,2)
model = ARIMA(x, order=(2, 0, 2))
result = model.fit()
print(result.summary())

ar1 = result.params[1]
ar2 = result.params[2]
ma1 = result.params[3]
ma2 = result.params[4]

se_ar1 = result.bse[1]
se_ar2 = result.bse[2]
se_ma1 = result.bse[3]
se_ma2 = result.bse[4]

# Intervalli di confidenza 95%
ci_ar1 = ar1 + np.array([-1.96, 1.96]) * se_ar1
ci_ar2 = ar2 + np.array([-1.96, 1.96]) * se_ar2
ci_ma1 = ma1 + np.array([-1.96, 1.96]) * se_ma1
ci_ma2 = ma2 + np.array([-1.96, 1.96]) * se_ma2

print("IC 95% AR(1):", ci_ar1)
print("IC 95% AR(2):", ci_ar2)
print("IC 95% MA(1):", ci_ma1)
print("IC 95% MA(2):", ci_ma2)


##########################################################
#       SCEGLIERE L'ORDINE p,g
#

import numpy as np
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(2)

# Simuliamo un processo ARMA(3,2) con coefficienti specificati
arparams = np.array([0.5, -0.25, 0.4])
maparams = np.array([0.5, -0.3])
ar = np.r_[1, -arparams]  # statsmodels usa coefficenti negativi per AR
ma = np.r_[1, maparams]
x = sm.tsa.arma_generate_sample(ar, ma, nsample=1000)

# Ora cerchiamo l'ordine ottimale (p, q) per un modello ARMA utilizzando il criterio AIC
best_aic = np.inf
best_order = None
best_model = None

# Proviamo tutte le combinazioni (p, q) con p e q da 0 a 4
for p, q in itertools.product(range(5), range(5)):
    try:
        model = sm.tsa.ARIMA(x, order=(p, 0, q)).fit()
        current_aic = model.aic
        if current_aic < best_aic:
            best_aic = current_aic
            best_order = (p, 0, q)
            best_model = model
    except:
        continue

# Stampiamo i risultati finali
print("AIC finale:", best_aic)
print("Ordine ARMA trovato:", best_order)
print(best_model.summary())

plot_acf(best_model.resid, lags=30)
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test
resid = best_model.resid
ljung_box_result = acorr_ljungbox(resid, lags=[20], return_df=True)
print(ljung_box_result)

##########################################################
#       S&P 500
#

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scarica i dati dell'S&P500
data = yf.download("^GSPC",period=max, group_by='ticker')
data = data["^GSPC"]

# Calcola i log-rendimenti giornalieri
log_returns = np.log(data['Close']).diff().dropna()

# Imposta la frequenza come giorni lavorativi
# log_returns = log_returns.asfreq('B')

import statsmodels.api as sm
import itertools

# Ricerca del modello ARMA(p,q) ottimale (con d=0) usando l'AIC
best_aic = np.inf
best_order = None
best_model = None

# Proviamo tutte le combinazioni (p, q) con p e q da 0 a 4
for p, q in itertools.product(range(5), range(5)):
    for q in range(5):
        try:
            model = sm.tsa.ARIMA(log_returns, order=(p, 0, q)).fit()
            current_aic = model.aic
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (p, 0, q)
                best_model = model
        except:
            continue

print("Ordine migliore:", best_order)

# Plot dell'ACF dei residui del modello ottimale
residuals = best_model.resid
sm.graphics.tsa.plot_acf(residuals.dropna(), lags=30)
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test
resid = best_model.resid
ljung_box_result = acorr_ljungbox(resid, lags=[20], return_df=True)
print(ljung_box_result)


