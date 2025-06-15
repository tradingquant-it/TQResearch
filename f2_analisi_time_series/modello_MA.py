"""
#############################################################
#                   MA(1)
#
"""
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Imposta il seed per la riproducibilità
np.random.seed(1)
N = 100
# Genera 100 rumori bianchi standard
w = np.random.normal(size=N)

# Inizializza la serie AR(1)
x = np.zeros(N)
for t in range(1, N):
    x[t] = w[t] + 0.6 * w[t - 1]

# Grafico modello AR
plt.plot(x)

# Grafico Autocorrelazione
plot_acf(x, lags=30)
plt.show()

"""
Come abbiamo visto nella formula per γₖ, per k > q tutte le autocorrelazioni dovrebbero risultare nulle. Dal momento
 che q = 1, ci aspettiamo un picco significativo per k = 1 e picchi non significativi per i valori successivi. 
 Tuttavia, a causa del bias campionario, possiamo aspettarci un 5% di picchi marginalmente significativi in un 
 grafico di autocorrelazione.

Il correlogramma mostra esattamente questo: un picco significativo per k = 1 e picchi non significativi per k > 1,
 eccetto per k = 4, dove osserviamo un picco marginalmente significativo.

Questo metodo è utile per verificare se un modello MA(q) è adatto. Osservando il correlogramma, possiamo contare 
quanti ritardi non nulli consecutivi esistono. Se troviamo q ritardi di questo tipo, possiamo tentare legittimamente 
di adattare un modello MA(q) alla serie.

Nel nostro caso, avendo simulato una serie MA(1), proveremo ora ad adattare un modello MA(1) ai dati simulati.

Sfortunatamente in Python (come in R) non esiste un comando specifico per i modelli MA come ar lo è per i modelli AR. 
Tuttavia, possiamo usare la funzione generale ARIMA impostando a zero i parametri autoregressivi e di integrazione.

"""
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(x, order=(0, 0, 1))
result = model.fit()
print(result.summary())

"""
Il comando ARIMA restituisce un output utile. Il parametro MA è stato stimato come φ̂₁ = 0.602, molto vicino al 
valore reale φ₁ = 0.6. Inoltre, ci fornisce gli errori standard, rendendo facile il calcolo degli intervalli di 
confidenza. Riceviamo anche la varianza stimata, il log-likelihood e l’AIC (criterio di informazione di Akaike) 
utile per confrontare modelli.

La differenza principale tra ARIMA e AR è che ARIMA stima anche un termine di intercetta perché non sottrae la 
media della serie. Dobbiamo quindi fare attenzione quando effettuiamo previsioni.

Calcolo dell’intervallo di confidenza per il parametro MA
Come controllo rapido, possiamo calcolare l’intervallo di confidenza al 95% per φ̂₁:
"""

phi_hat = result.params[1]
print(phi_hat)

se = result.bse[1]
print(se)

conf_int = phi_hat + np.array([-1.96, 1.96]) * se
print(conf_int)

"""
L’intervallo di confidenza al 95% contiene il valore vero φ₁ = 0.6, quindi possiamo considerare il modello 
una buona approssimazione. Ovviamente questo è atteso, dato che abbiamo simulato noi stessi i dati.
"""

#############################################################
#                   MA(1)  negativo
#

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(1)
w = np.random.normal(size=100)
x = np.zeros(100)

for t in range(1, 100):
    x[t] = w[t] - 0.6 * w[t-1]

plt.plot(x)
plot_acf(x)
plt.show()

"""
Possiamo osservare che per k=1 abbiamo un picco significativo nel correlogramma, ma questa volta mostra 
una correlazione negativa, come ci aspetteremmo da un modello MA(1) con coefficiente negativo. 
Anche in questo caso, tutti i picchi per k>1 sono insignificanti.

Adesso proviamo ad adattare un modello MA(1) e stimare il parametro:
"""

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(x, order=(0, 0, 1))
result = model.fit()
print(result.summary())

"""
Il parametro stimato è 𝜙^1 =−0.730, una lieve sottostima rispetto al valore vero 𝜙1 =−0.6.

Calcoliamo infine l’intervallo di confidenza al 95%:
"""

phi_hat = result.params[1]
print(phi_hat)

se = result.bse[1]
print(se)

conf_int = phi_hat + np.array([-1.96, 1.96]) * se
print(conf_int)

"""
Possiamo vedere che il valore vero 𝜙1=−0.6 è contenuto nell’intervallo di confidenza al 95%, confermando così
che il modello si adatta bene ai dati.
"""

#############################################################
#                   MA(3)
#

"""
MA(3)
Possiamo seguire la stessa procedura per un processo MA(3). In questo caso, ci aspettiamo picchi significativi 
per k∈{1,2,3} e picchi non significativi per k>3.

Utilizzeremo i seguenti coefficienti: 𝜃1=0,6, 𝜃2=0,4 e 𝜃3=0,3. Simuliamo quindi un processo MA(3) basato su 
questo modello. In questa simulazione aumentiamo il numero di campioni casuali a 1000, il che rende più 
evidente la struttura dell'autocorrelazione, anche se la serie originale risulta più difficile da interpretare.
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(3)
n = 1000
w = np.random.normal(size=n)
x = w.copy()

for t in range(3, n):
    x[t] = w[t] + 0.6*w[t-1] + 0.4*w[t-2] + 0.3*w[t-3]

plt.plot(x)
plot_acf(x)
plt.show()

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(x, order=(0, 0, 3))
result = model.fit()
print(result.summary())

"""
L'output fornisce le stime dei parametri: 𝜃^1=0,544  𝜃^2=0,345  𝜃^3=0,298
Questi valori sono vicini ai veri parametri 𝜃1=0,6, 𝜃2=0,4 e 𝜃3=0,3. Possiamo anche calcolare gli intervalli 
di confidenza al 95% usando gli errori standard:
"""

# Estrazione dei parametri e degli errori standard
params = result.params
stderr = result.bse

# Intervalli di confidenza al 95% per i parametri MA
conf_int_theta1 = params[1] + np.array([-1.96, 1.96]) * stderr[1]
conf_int_theta2 = params[2] + np.array([-1.96, 1.96]) * stderr[2]
conf_int_theta3 = params[3] + np.array([-1.96, 1.96]) * stderr[3]

print("IC 95% θ1:", conf_int_theta1)
print("IC 95% θ2:", conf_int_theta2)
print("IC 95% θ3:", conf_int_theta3)


