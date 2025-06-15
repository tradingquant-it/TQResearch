"""
Indice azionario S&P500 USA

Oltre ai singoli titoli, possiamo anche considerare l’indice azionario statunitense S&P500. Applichiamo tutti i
comandi utilizzati in precedenza a questa serie e generiamo i grafici come fatto prima:
"""

import yfinance as yf
import pandas as pd

# Scarica i dati di SP500
symbol = "^GSPC"
sp500 = yf.download(symbol, start="2007-01-01", end="2015-08-15",group_by='ticker')
sp500 = sp500[symbol]

sp500.index = pd.to_datetime(sp500.index)
sp500 = sp500.asfreq('B').dropna()

print(sp500)
"""
Possiamo ora rappresentare graficamente i prezzi di chiusura, 
"""

import matplotlib.pyplot as plt

sp500['Close'].plot(title="sp500 Daily Closing Price")
plt.xlabel("Data")
plt.ylabel("Prezzo ($)")
plt.grid(True)
plt.show()

"""
Ora calcoliamo i rendimenti logaritmici di sp500 e la loro differenza al primo ordine, trasformando così la serie
 dei prezzi da non stazionaria a una (potenzialmente) stazionaria.
"""
import numpy as np
# Calcola rendimenti logaritmici e poi differenza al primo ordine
log_returns = np.log(sp500['Close'])
sp500rt = log_returns.diff().dropna()

"""
Visualizziamo la serie differenziata con un grafico:
"""

plt.plot(sp500rt)
plt.title("Log-rendimenti differenziati di sp500")
plt.show()
"""
È chiaro dal grafico che la volatilità non è stazionaria nel tempo. Questo risulta evidente anche dal correlogramma, 
come mostrato nella Figura 10.9. Si notano molti picchi, inclusi quelli per k = 1 e k = 2, che risultano 
statisticamente significativi, il che va oltre quanto previsto da un semplice modello di white noise.

Inoltre, osserviamo segni di processi a memoria lunga, con alcuni picchi significativi anche per k = 16, k = 18 
e k = 21:
"""

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(sp500rt, lags=30)
plt.show()
print("")

"""
Alla fine sarà necessario un modello più sofisticato di un semplice modello autoregressivo di ordine p. 
Tuttavia, in questa fase possiamo comunque provare a stimare un modello AR e osservare i risultati:
"""
from statsmodels.tsa.ar_model import ar_select_order

# Stima l'ordine p ottimale con criterio AIC (come in R)
selected = ar_select_order(sp500rt.dropna(), maxlag=30, ic='aic')

# Fit finale del modello AR con l'ordine ottimale
model = selected.model
results = model.fit()

# Estrazione dei parametri e della matrice di varianza-covarianza
params = results.params
cov_matrix = results.cov_params()

# Ordine stimato e parametri
print("Ordine AR stimato:", results.model._lags)
print("Parametri stimati:", params)

"""
L’uso del comando AutoReg ha prodotto un modello AR(22), ovvero con 22 parametri diversi da zero. Cosa ci dice questo 
risultato? Ci suggerisce che esiste una complessità significativa nella correlazione seriale che un semplice modello 
lineare dei prezzi passati non riesce a spiegare pienamente.

Ma in fondo lo sapevamo già: osservando il grafico dei rendimenti, emerge una forte serial correlation nella 
volatilità, ad esempio nel periodo altamente turbolento intorno al 2008.

Questo motiva l’introduzione della prossima classe di modelli: i Moving Average MA(q) e i modelli ARMA(p, q). 
Analizzeremo entrambi nelle prossime sezioni di questo capitolo.

Come sottolineato più volte, questi modelli ci condurranno infine verso le famiglie ARIMA e GARCH, in grado di 
fornire un adattamento molto più efficace alla complessità seriale dell’indice S&P500.

Questo ci permetterà di migliorare significativamente le previsioni e, in definitiva, di sviluppare strategie 
più redditizie.
"""