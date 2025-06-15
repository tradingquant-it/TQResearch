
import numpy as np

# Imposta il seed per la riproducibilità
np.random.seed(1)

# Genera 100 rumori bianchi standard
w = np.random.normal(size=100)

# Inizializza la serie AR(1)
simulated_data_1 = np.copy(w)
for t in range(1, 100):
    simulated_data_1[t] = 0.6 * simulated_data_1[t - 1] + w[t]


# Alternativa usando la libreria statsmodel 

# from statsmodels.tsa.arima_process import ArmaProcess

# ar1 = np.array([1, -0.6])
# ma1 = np.array([1])
# AR_object1 = ArmaProcess(ar1, ma1)
# simulated_data_1 = AR_object1.generate_sample(nsample=100)


from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Grafico modello AR
plt.plot(simulated_data_1)

# Grafico Autocorrelazione
plot_acf(simulated_data_1, lags=30)
plt.show()



#############################################################
#                   AR(1)
#

import numpy as np
np.random.seed(1)
w = np.random.normal(size=100)
x = np.copy(w)

for t in range(1, 100):
    x[t] = 0.6 * x[t-1] + w[t]

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(x, order=(1, 0, 0))
result = model.fit()
print(result.model_orders)
print("")


phi_hat = result.params[1]
print(phi_hat)


se = result.bse[1]
conf_int = phi_hat + np.array([-1.96, 1.96]) * se
print(conf_int)

####

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(1)
w = np.random.normal(size=100)
x = np.copy(w)

for t in range(1, 100):
    x[t] = -0.6 * x[t-1] + w[t]

plt.plot(x)
plot_acf(x)
plt.show()


#
#   Alpha = -0,6
#

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Riproducibilità
np.random.seed(1)
N = 1000
w = np.random.normal(size=N)
x = np.copy(w)

# Generazione del processo AR(1) con phi = -0.6
for t in range(1, N):
    x[t] = -0.6 * x[t - 1] + w[t]

# Modello AR(1) senza intercetta (trend='n' => nessuna costante)
model = ARIMA(x, order=(1, 0, 0), trend='n')
result = model.fit()

# Estrazione dei parametri e intervallo di confidenza
import pandas as pd

params = pd.Series(result.params, index=result.param_names)
bse = pd.Series(result.bse, index=result.param_names)

phi_hat = params['ar.L1']
se = bse['ar.L1']
conf_int = phi_hat + np.array([-1.96, 1.96]) * se

# Output
print("Ordine stimato:", result.model_orders)
print("Stima di phi_1:", phi_hat)
print("Intervallo di confidenza al 95%:", conf_int)


#############################################################
#                   AR(2)
#

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(1)
N = 100
w = np.random.normal(size=N)
x = w.copy()

for t in range(2, N):
    x[t] = 0.666 * x[t-1] - 0.333 * x[t-2] + w[t]

plt.plot(x)
plot_acf(x)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

np.random.seed(1)
N = 1000
w = np.random.normal(size=N)
x = w.copy()

for t in range(2, N):
    x[t] = 0.666 * x[t-1] - 0.333 * x[t-2] + w[t]

model = ARIMA(x, order=(2, 0, 0))
result = model.fit()

print("Ordine stimato:", result.model_orders)
print("Stime dei parametri AR:", result.params[1:3])
