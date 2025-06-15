
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

np.random.seed(123)

z = np.zeros(10000)
for i in range(1, 10000):
    z[i] = z[i-1] + np.random.normal()

p = 0.3 * z + np.random.normal(size=10000)
q = 0.6 * z + np.random.normal(size=10000)
r = 0.2 * z + np.random.normal(size=10000)


data = pd.DataFrame({'p': p, 'q': q, 'r': r})
jotest = coint_johansen(data, det_order=-1, k_ar_diff=2)

# Stampa dei risultati
print("######################")
print("# Procedura Johansen #")
print("######################")
print("Tipo di test: statistica trace, senza termine costante")
print("Autovalori (lambda):")
print(jotest.eig)
print("\nValori della statistica di test e valori critici:")
for idx, label in enumerate([0,1,2]):
    stat = jotest.lr1[idx]
    crit = jotest.cvt[idx]     # [10%, 5%, 1%]
    print(f"H0: r <= {label} | test = {stat:.2f}, 10% = {crit[0]:.2f}, 5% = {crit[1]:.2f}, 1% = {crit[2]:.2f}")

print("\nAutovettori normalizzati (relazioni di cointegrazione):")
print(pd.DataFrame(jotest.evec, columns=['p.l2', 'q.l2', 'r.l2']))

print("\nPesi W (matrice di caricamento):")
print(pd.DataFrame(jotest.eig, columns=["W"]))

import matplotlib.pyplot as plt

# Supponendo che p, q e r siano array o pandas Series già definiti
s = 0.121041 * p - 0.598822 * q + 1.614455 * r

plt.plot(s)
plt.show()


from arch.unitroot import ADF

adf = ADF(s)
print(adf.summary())

###############################################################
#           ETF - EWA e EWC
#

import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

EWA = yf.download('EWA', start="2006-04-26", end="2012-04-09", group_by='ticker', auto_adjust=False)
EWA = EWA['EWA']
EWC = yf.download('EWC', start="2006-04-26", end="2012-04-09", group_by='ticker', auto_adjust=False)
EWC = EWC['EWC']
IGE = yf.download('IGE', start="2006-04-26", end="2012-04-09", group_by='ticker', auto_adjust=False)
IGE = IGE['IGE']

df = pd.DataFrame({'EWA': EWA['Adj Close'], 'EWC': EWC['Adj Close'], 'IGE': IGE['Adj Close']})

jotest = coint_johansen(df, 0, 1)


# Stampa dei risultati
print("######################")
print("# Procedura Johansen #")
print("######################")
print("Tipo di test: statistica trace, senza termine costante")
print("Autovalori (lambda):")
print(jotest.eig)
print("\nValori della statistica di test e valori critici:")
for idx, label in enumerate([0,1,2]):
    stat = jotest.lr1[idx]
    crit = jotest.cvt[idx]     # [10%, 5%, 1%]
    print(f"H0: r <= {label} | test = {stat:.2f}, 10% = {crit[0]:.2f}, 5% = {crit[1]:.2f}, 1% = {crit[2]:.2f}")

print("\nAutovettori normalizzati (relazioni di cointegrazione):")
print(pd.DataFrame(jotest.evec, columns=['p.l2', 'q.l2', 'r.l2']))

print("\nPesi W (matrice di caricamento):")
print(pd.DataFrame(jotest.eig, columns=["W"]))


###############################################################
#           SPY IVV VOO
#

import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

SPY = yf.download('SPY', start="2016-01-01", end="2016-12-31", group_by='ticker', auto_adjust=False)
SPY = SPY['SPY']
IVV = yf.download('IVV', start="2016-01-01", end="2016-12-31", group_by='ticker', auto_adjust=False)
IVV = IVV['IVV']
VOO = yf.download('VOO', start="2016-01-01", end="2016-12-31", group_by='ticker', auto_adjust=False)
VOO = VOO['VOO']

df = pd.DataFrame({'SPY': SPY['Adj Close'], 'IVV': IVV['Adj Close'], 'VOO': VOO['Adj Close']})

jotest = coint_johansen(df, 0, 1)


# Stampa dei risultati
print("######################")
print("# Procedura Johansen #")
print("######################")
print("Tipo di test: statistica trace, senza termine costante")
print("Autovalori (lambda):")
print(jotest.eig)
print("\nValori della statistica di test e valori critici:")
for idx, label in enumerate([0,1,2]):
    stat = jotest.lr1[idx]
    crit = jotest.cvt[idx]     # [10%, 5%, 1%]
    print(f"H0: r <= {label} | test = {stat:.2f}, 10% = {crit[0]:.2f}, 5% = {crit[1]:.2f}, 1% = {crit[2]:.2f}")

print("\nAutovettori normalizzati (relazioni di cointegrazione):")
print(pd.DataFrame(jotest.evec, columns=['p.l2', 'q.l2', 'r.l2']))

print("\nPesi W (matrice di caricamento):")
print(pd.DataFrame(jotest.eig, columns=["W"]))
