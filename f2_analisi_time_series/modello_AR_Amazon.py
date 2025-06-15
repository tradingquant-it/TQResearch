"""
10.4.5 Dati Finanziari
Amazon Inc.

Iniziamo recuperando il prezzo delle azioni Amazon (ticker: AMZN), utilizzando la libreria yfinance in Python,
 che svolge un ruolo simile a quantmod in R:
"""

import yfinance as yf
import pandas as pd

# Scarica i dati di Amazon (AMZN)
symbol = "AMZN"
amzn = yf.download(symbol, start="2007-01-01", end="2015-08-15",group_by='ticker')
amzn = amzn[symbol]

amzn.index = pd.to_datetime(amzn.index)
amzn = amzn.asfreq('B').dropna()

print(amzn)
"""
L’output mostra le quotazioni giornaliere, inclusi apertura, massimo, minimo, chiusura, volume e chiusura aggiustata:

Il primo passo consiste sempre nel visualizzare rapidamente il prezzo per una prima ispezione grafica. In questo 
esempio, utilizziamo i prezzi di chiusura giornalieri:
"""

import matplotlib.pyplot as plt

amzn['Close'].plot(title="Amazon Daily Closing Price")
plt.xlabel("Data")
plt.ylabel("Prezzo ($)")
plt.grid(True)
plt.show()

"""
La libreria yfinance formatta automaticamente l’asse delle date e produce un grafico chiaro e leggibile, molto 
simile a quanto visto con quantmod in R.

Ora calcoliamo i rendimenti logaritmici di AMZN e la loro differenza al primo ordine, trasformando così la serie
 dei prezzi da non stazionaria a una (potenzialmente) stazionaria.

Questo passaggio è utile per confrontare in modo coerente azioni, indici o altri asset, e permette di effettuare 
analisi multivariate, come il calcolo della matrice di covarianza. Creiamo una nuova serie amznrt con i 
log-rendimenti differenziati:
"""
import numpy as np
# Calcola rendimenti logaritmici e poi differenza al primo ordine
log_returns = np.log(amzn['Close'])
amznrt = log_returns.diff().dropna()

"""
Visualizziamo la serie differenziata con un grafico:
"""

plt.plot(amznrt)
plt.title("Log-rendimenti differenziati di AMZN")

"""
A questo punto vogliamo osservare il correlogramma per verificare se la serie differenziata si comporta come un 
rumore bianco. Se rileviamo autocorrelazione, potremmo modellarla con un AR. Vediamo il grafico ACF:
"""

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(amznrt, lags=20)
plt.show()
print("")

"""
Notiamo un picco significativo a lag 𝑘 = 2, il che suggerisce autocorrelazione residua. Questo potrebbe essere 
causato dal campionamento, ma possiamo comunque adattare un modello AR(p) alla serie e stimare gli intervalli 
di confidenza dei parametri:
"""
from statsmodels.tsa.ar_model import AutoReg

# Adattiamo un modello AR automatico
model = AutoReg(amznrt, lags=2, old_names=False)
results = model.fit()

# Estrazione dei parametri e della matrice di varianza-covarianza
params = results.params
cov_matrix = results.cov_params()

# Ordine stimato e parametri
print("Ordine AR stimato:", results.model._lags)
print("Parametri stimati:", params)
print("Varianza asintotica dei parametri:")
print(cov_matrix)
# print(results.bse ** 2)

"""
L'adattamento del modello autoregressivo AR alla serie di log-prezzi differenziata al primo ordine restituisce un 
modello AR(2), con stime dei parametri 𝜙^1 = −0,0278 e 𝜙^2=−0,0687. Abbiamo anche stampato la varianza asintotica
per poter calcolare gli errori standard dei parametri e costruire gli intervalli di confidenza. V
ogliamo verificare se lo zero rientra nell'intervallo di confidenza al 95%, poiché ciò influenzerebbe la nostra 
fiducia nel considerare l’AR(2) come vero modello sottostante per la serie AMZN.

Per calcolare gli intervalli di confidenza al 95% per ciascun parametro, utilizziamo i comandi seguenti in Python. 
Calcoliamo la radice quadrata dell’elemento diagonale della matrice di varianza-covarianza per ottenere 
l’errore standard, quindi moltiplichiamo per -1.96 e 1.96 per ottenere i limiti dell’intervallo:
"""

# Calcolo intervallo di confidenza al 95% per phi1 e phi2
se_phi1 = np.sqrt(cov_matrix.iloc[1, 1])
ci_phi1 = params.iloc[1] + np.array([-1.96, 1.96]) * se_phi1
print("Intervallo di confidenza per φ1:", ci_phi1)

se_phi2 = np.sqrt(cov_matrix.iloc[2, 2])
ci_phi2 = params.iloc[2] + np.array([-1.96, 1.96]) * se_phi2
print("Intervallo di confidenza per φ2:", ci_phi2)


"""
Si nota che lo zero rientra nell’intervallo di confidenza per 𝜙1, ma non per 𝜙2. Pertanto, dobbiamo essere cauti 
nel concludere che esista un vero processo generativo AR(2) alla base della serie AMZN.

È importante osservare che il modello autoregressivo non considera il volatility clustering, ovvero 
l’accorpamento della volatilità che spesso caratterizza le serie finanziarie. Questo effetto comporta una 
correlazione seriale raggruppata, che i modelli ARCH e GARCH (che affronteremo nei prossimi capitoli) 
possono modellare in modo più accurato.

Quando utilizzeremo la funzione completa ARIMA nella sezione dedicata alle strategie di trading, 
impiegheremo le previsioni sui log-prezzi giornalieri per generare segnali operativi.
"""