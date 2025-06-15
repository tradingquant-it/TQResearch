import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance
import hmmlearn

np.random.seed(1)

# Create the parameters for the bull and bear market returns distributions
Nk_lower = 50
Nk_upper = 150
bull_mean = 0.1
bull_var = 0.1
bear_mean = -0.05
bear_var = 0.2

# Create the list of durations (in days) for each regime
days = np.random.randint(Nk_lower, Nk_upper, 5)

# Create the various bull and bear markets returns
market_bull_1 = np.random.normal(loc=bull_mean, scale=bull_var, size=days[0])
market_bear_2 = np.random.normal(loc=bear_mean, scale=bear_var, size=days[1])
market_bull_3 = np.random.normal(loc=bull_mean, scale=bull_var, size=days[2])
market_bear_4 = np.random.normal(loc=bear_mean, scale=bear_var, size=days[3])
market_bull_5 = np.random.normal(loc=bull_mean, scale=bull_var, size=days[4])

# Creazione dei veri stati di regime: 1 = bullish, 2 = bearish
true_regimes = np.concatenate([np.ones(days[0]), np.full(days[1], 2), np.ones(days[2]),
                               np.full(days[3], 2), np.ones(days[4])])

returns = np.concatenate([market_bull_1, market_bear_2, market_bull_3, market_bear_4, market_bull_5])

plt.plot(returns)
plt.show()

from hmmlearn.hmm import GaussianHMM

# Reshape dei dati per hmmlearn
X = returns.reshape(-1, 1)

# Definizione e training del modello HMM
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
hmm.fit(X)

# Estrazione delle probabilità posteriori
post_probs = hmm.predict_proba(X)
predicted_states = hmm.predict(X)

fig, axs = plt.subplots(2) # Definezione di due grafici
axs[0].plot(predicted_states)
axs[0].set(ylabel='Regimes')
axs[1].plot(post_probs, label=['Bull','Bear'])
axs[1].set(ylabel='Probability')
axs[1].legend()
plt.show()



#############################################################
#          S&P500
#


import matplotlib.pyplot as plt
import yfinance


data = yfinance.download("^GSPC", start="2004-01-01", end="2017-11-01", group_by='ticker', auto_adjust=False)
data = data["^GSPC"]
data = data.asfreq('b').fillna(method='ffill')

Return = data['Adj Close'].pct_change()
LogRet = np.log(data['Adj Close']).diff().dropna()

#plt.plot(LogRet)
#plt.show()


from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' se disponibile

rets = np.column_stack([LogRet])
# Create the Gaussian Hidden markov Model and fit it to the returns data
hmm_model = GaussianHMM(
    n_components=2, covariance_type="full", n_iter=1000
).fit(rets)
post_prob = hmm_model.predict_proba(rets)

# Creiamo il grafico con dimensioni esplicite
fig, axs = plt.subplots(2, figsize=(12, 6), sharex=True)

axs[0].plot(rets, color='blue')
axs[0].set_ylabel('Returns')

axs[1].plot(post_prob, label=['Regime 1', 'Regime 2'])
axs[1].set_ylabel('Probability')
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.show()


# 3 STATI

from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' se disponibile

rets = np.column_stack([LogRet])
# Create the Gaussian Hidden markov Model and fit it to the returns data
hmm_model = GaussianHMM(
    n_components=3, covariance_type="full", n_iter=1000
).fit(rets)
post_prob = hmm_model.predict_proba(rets, None)

# Creiamo il grafico con dimensioni esplicite
fig, axs = plt.subplots(2, figsize=(12, 6), sharex=True)

axs[0].plot(rets, color='blue')
axs[0].set_ylabel('Returns')

axs[1].plot(post_prob, label=['Regime 1', 'Regime 2','Regime 3'])
axs[1].set_ylabel('Probability')
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
