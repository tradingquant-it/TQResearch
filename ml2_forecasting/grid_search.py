
# Codice della lezione "Ottimizzazione dei Parametri nel Trading Algoritmico"

# grid_search.py

import datetime

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVC
from forecast import create_lagged_series


if __name__ == "__main__":
    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series(
      "^GSPC",
      datetime.datetime(2001,1,10),
      datetime.datetime(2005,12,31), lags=5
    )

    # Use the prior two days of returns as predictor
    # values, with direction as the response
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Set the parameters by cross-validation
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    ]

    # Perform the grid search on the tuned parameters
    model = GridSearchCV(SVC(C=1), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    print("Optimised parameters found on training set:")
    print(model.best_estimator_, "\n")
    print("Grid scores calculated on training set:")
    results = model.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print("%0.3f for %r" % (mean_score, params))
