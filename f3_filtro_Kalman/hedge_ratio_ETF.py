import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pykalman import KalmanFilter


def draw_date_coloured_scatterplot(etfs, prices):
    """
    Creare un grafico scatterplot dei prezzi di due ETF, che è
    colorato dalle date dei prezzi per indicare il cambiamento
    della relazione tra le due serie di prezzi
    """

    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)

    # Creare l'oggetto scatterplot
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
    )

    # Aggiungere una barra di colori per la colorazione dei dati ed
    # impostare le etichette dell'asse corrispondente
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen // 9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()


def calc_slope_intercept_kalman(etfs, prices):
    """
     Utilizzo del filtro Kalman dal pacchetto pyKalman
     per calcolare la pendenza e l'intercetta della
     regressione lineare dei prezzi degli ETF.
     """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs

def draw_slope_intercept_changes(prices, state_means):
    """
    Tracciare la variazione di pendenza e intercetta
    dai valori calcolati dal Filtro di Kalman.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0],
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()

if __name__ == "__main__":
    # Scegliere i simboli ETF symbols e il periodo temporale
    # dei prezzi storici
    etfs = ['TLT', 'IEI']
    start_date = "2012-10-01"
    end_date = "2017-10-01"

    # Download dei prezzi di chiusura da Yahoo finance
    prices = yf.download(etfs, start=start_date, end=end_date, auto_adjust=False)
    prices = prices['Adj Close']

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)