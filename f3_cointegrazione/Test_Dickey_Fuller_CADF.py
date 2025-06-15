
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

np.random.seed(123)
n = 1000

z = np.zeros(n)
for i in range(1, n):
    z[i] = z[i-1] + np.random.standard_normal(1)

p = 0.3 * z + np.random.standard_normal(1000)
q = 0.6 * z + np.random.standard_normal(1000)

q_const = sm.add_constant(q)

comb = sm.OLS(p, q_const)
model = comb.fit()
print(model.summary())

from arch.unitroot import ADF

resid = model.resid
adf = ADF(resid)
print(adf.summary())


###############################################################
#           ETF - EWA e EWC
#

import yfinance as yf

EWA = yf.download('EWA', start="2006-04-26", end="2012-04-09", group_by='ticker', auto_adjust=False)
EWA = EWA['EWA']

EWC = yf.download('EWC', start="2006-04-26", end="2012-04-09", group_by='ticker', auto_adjust=False)
EWC = EWC['EWC']

ewa = EWA['Adj Close']
ewc = EWC['Adj Close']


import matplotlib.pyplot as plt

plt.plot(ewa.values, color="blue")
plt.plot(ewc.values, color="red")
plt.show()

plt.scatter(ewa, ewc)
plt.show()

import statsmodels.api as sm

comb1 = sm.OLS(ewc, sm.add_constant(ewa)).fit()
comb2 = sm.OLS(ewa, sm.add_constant(ewc)).fit()

resid1 = comb1.resid
plt.plot(resid1)
plt.show()

print(comb1.summary())

print(comb2.summary())


from arch.unitroot import ADF

adf1 = ADF(comb1.resid, lags=1)
print(adf1.summary())

adf2 = ADF(comb2.resid, lags=1)
print(adf2.summary())
