# ensemble_prediction.py

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import sklearn
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor

def create_lagged_series(symbol, start_date, end_date, lags=3):
    """
    Crea un DataFrame panda che memorizza
    i rendimenti percentuali dell valore della chiusura
    rettificata di un assest scaricato da Yahoo Finance,
    insieme a una serie di rendimenti ritardati dei
    giorni di trading precedenti (il ritardo predefinito è 3 giorni).
    È incluso anche il volume degli scambi.
    """

    # Scaricare i dati storici da Yahoo Finance
    ts = yf.download(symbol, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    ts = ts[symbol]

    # Creazione di un DataFrame dei ritardi
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Creazione della serie dei ritardi dei
    # prezzi di chiusura dei giorni precedenti
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Creazione del DataFrame dei rendimenti
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # Creazione delle colonne delle percentuali dei rendimenti ritardi
    for i in range(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag[
            "Lag%s" % str(i+1)
        ].pct_change()*100.0
    tsret = tsret[tsret.index >= start_date]
    return tsret

# Impostazione del seed random, numero di stimatori
# and lo "step factor" usato per il grafico di MSE
# per ogni metodo
random_state = 42
n_jobs = 1  # Fattore di parallelizazione per il bagging e random forests
n_estimators = 1000
step_factor = 10
axis_step = int(n_estimators/step_factor)

# Scaricare 10 anni di storico di Amazon
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2015, 12, 31)
amzn = create_lagged_series("AMZN", start, end, lags=3)
amzn.dropna(inplace=True)

# Uso dei ritardi dei primi 3 giorni dei prezzi close di AMZN
# e ridimensione dei dati ttra -1 e +1 per i confronti
X = amzn[["Lag1", "Lag2", "Lag3"]]
y = amzn["Today"]
X = scale(X)
y = scale(y)

# Divisione in training-testing con il 70% dei dati per il
# training e il rimanente 30% dei dati per il testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# Inizializzazione degli array che conterrano il
# MSE per ogni metodo d'insieme
estimators = np.zeros(axis_step)
bagging_mse = np.zeros(axis_step)
rf_mse = np.zeros(axis_step)
boosting_mse = np.zeros(axis_step)

# Stimare il Bagging MSE per l'intero numero di
# stimatore, con un passo specifico ("step_factor")
for i in range(0, axis_step):
    print("Bagging Estimator: %d of %d..." % (
        step_factor * (i + 1), n_estimators)
          )
    bagging = BaggingRegressor(
        DecisionTreeRegressor(),
        n_estimators=step_factor * (i + 1),
        n_jobs=n_jobs,
        random_state=random_state
    )
    bagging.fit(X_train, y_train)
    mse = mean_squared_error(y_test, bagging.predict(X_test))
    estimators[i] = step_factor * (i + 1)
    bagging_mse[i] = mse

# Stima del Random Forest MSE per l'intero numero di
# stimatori, con un passo specifico ("step_factor")
for i in range(0, axis_step):
    print("Random Forest Estimator: %d of %d..." % (
        step_factor * (i + 1), n_estimators)
          )
    rf = RandomForestRegressor(
        n_estimators=step_factor * (i + 1),
        n_jobs=n_jobs,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, rf.predict(X_test))
    estimators[i] = step_factor * (i + 1)
    rf_mse[i] = mse

# Stima del AdaBoost MSE per l'intero numero di
# stimatori, con un passo specifico ("step_factor")
for i in range(0, axis_step):
    print("Boosting Estimator: %d of %d..." % (
        step_factor * (i + 1), n_estimators)
          )
    boosting = AdaBoostRegressor(
        DecisionTreeRegressor(),
        n_estimators=step_factor * (i + 1),
        random_state=random_state,
        learning_rate=0.01
    )
    boosting.fit(X_train, y_train)
    mse = mean_squared_error(y_test, boosting.predict(X_test))
    estimators[i] = step_factor * (i + 1)
    boosting_mse[i] = mse

# Visualizzazione del grafico del MSE per il numero di stimatori
plt.figure(figsize=(8, 8))
plt.title('Bagging, Random Forest and Boosting comparison')
plt.plot(estimators, bagging_mse, 'b-', color="black", label='Bagging')
plt.plot(estimators, rf_mse, 'b-', color="blue", label='Random Forest')
plt.plot(estimators, boosting_mse, 'b-', color="red", label='AdaBoost')
plt.legend(loc='upper right')
plt.xlabel('Estimators')
plt.ylabel('Mean Squared Error')
plt.show()
print("")
