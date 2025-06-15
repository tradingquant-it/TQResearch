# perc_diabetes_sklearn.py

import pandas as pd
from sklearn.linear_model import Perceptron


if __name__ == "__main__":
    # Carica il set di dati sul diabete dal CSV e lo converte
    # in una matrice NumPy adatta per l'estrazione nel formato
    # X, y,  necessario per Scikit-Learn
    diabetes = pd.read_csv('diabetes.csv').values

    # Estrarre le colonne delle features e della risposta
    # del risultato nelle variabili appropriate
    X = diabetes[:, 0:8]
    y = diabetes[:, 8]

    # Crea e addestra un modello perceptron model (con un seed
    # random e riproducibile)
    model = Perceptron(random_state=1)
    model.fit(X, y)

    # Output il punteggio medio di precisione
    # della classificazione
    print("%0.3f" % model.score(X, y))

