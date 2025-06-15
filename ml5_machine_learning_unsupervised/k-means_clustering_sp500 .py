
import copy
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import (
    DateFormatter, WeekdayLocator, DayLocator, MONDAY
)
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans

def get_open_normalised_prices(data):
    """
    Restituisce un DataFrame pandas contenenti i prezzi high, low
    e close normalizzati con il prezzo open per i dati OHLC.
    Cioè si crea le colonne High/Open, Low/Open e Close/Open
    """
    df = data.copy()
    df["H/O"] = df["High"]/df["Open"]
    df["L/O"] = df["Low"]/df["Open"]
    df["C/O"] = df["Close"]/df["Open"]
    df.drop(
        [
            "Open", "High", "Low",
            "Close", "Volume", "Adj Close"
        ],
        axis=1, inplace=True
    )
    return df

def plot_candlesticks(data, since):
    """
    Visualizzo il grafico a candele dei prezzi,
    con specifica formatazzione delle date
    """
    # Copio e resetto l'indice del dataframe
    # per usare solo un sottoinsieme dei dati
    df = copy.deepcopy(data)
    df = df[df.index >= since]
    df.reset_index(inplace=True)
    df['date_fmt'] = df['Date'].apply(
        lambda date: mdates.date2num(date.to_pydatetime())
    )

    # Imposto la corretta formattazione dell'asse delle date
    # con Lunedi evidenziato come un punto principale
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d')
    fig, ax = plt.subplots(figsize=(16,4))
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Visualizzo il grafico a candele OHLC, dove i giorni positivi
    # sono candele nere e quelli negativi sono rosse
    csticks = candlestick_ohlc(
        ax, df[
            ['date_fmt', 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6,
        colorup='#000000', colordown='#ff0000'
    )
    ax.set_facecolor((1,1,0.9))
    ax.xaxis_date()
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45, horizontalalignment='right'
    )
    plt.show()

def plot_3d_normalised_candles(data, labels):
    """
    Visualizza un grafico a dispersione 3D di tutte le candelle
    normalizzate con l'apertura evidenziando i cluster con colori diversi
    """
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig, elev=21, azim=-136)
    ax.scatter(
        data["H/O"], data["L/O"], data["C/O"],
        c=labels.astype(float)
    )
    ax.set_xlabel('High/Open')
    ax.set_ylabel('Low/Open')
    ax.set_zlabel('Close/Open')
    plt.show()


def plot_cluster_ordered_candles(data):
    """
    Visualizza il grafico a candele ordinato secondo l'appartenza
    ad un cluster con la linea tratteggiata blu che rappresenta
    il bordo di ogni cluster.
    """
    # Imposto il formato per gli assi per formmattare
    # correttamente le date, con Lunedi come tick principale
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter("")
    fig, ax = plt.subplots(figsize=(16,4))
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Ordino i dati in base ai valori dei cluster e ottengo
    # un dataframe che contiene i valori degli indici per
    # ogni variazione dei bordi dei cluster
    df = copy.deepcopy(data)
    df.sort_values(by="Cluster", inplace=True)
    df.reset_index(inplace=True)
    df["clust_index"] = df.index
    df["clust_change"] = df["Cluster"].diff()
    change_indices = df[df["clust_change"] != 0]

    # Visualizzo il grafico OHLC con le candele ordinati in base ai cluster
    csticks = candlestick_ohlc(
        ax, df[
            ["clust_index", 'Open', 'High', 'Low', 'Close']
        ].values, width=0.6,
        colorup='#000000', colordown='#ff0000'
    )
    ax.set_facecolor((1,1,0.9))

    # Aggiungo ogni bordi di cluster come una linea blu trattegiata
    for row in change_indices.iterrows():
        plt.axvline(
            row[1]["clust_index"],
            linestyle="dashed", c="blue"
        )
    plt.xlim(0, len(df))
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45, horizontalalignment='right'
    )
    plt.show()


def create_follow_cluster_matrix(data):
    """
    Crea una matrice k x k, dove k è il numero di clusters
    che mostra quanto il cluster i è seguito dal cluster j.
    """
    data["ClusterTomorrow"] = data["Cluster"].shift(-1)
    data.dropna(inplace=True)
    data["ClusterTomorrow"] = data["ClusterTomorrow"].apply(int)
    data["ClusterMatrix"] = list(zip(data["Cluster"], data["ClusterTomorrow"]))
    cmvc = data["ClusterMatrix"].value_counts()
    clust_mat = np.zeros( (k, k) )
    for row in cmvc.items():
        clust_mat[row[0]] = row[1]*100.0/len(data)
    print("Cluster Follow-on Matrix:")
    print(clust_mat)


if __name__ == "__main__":
    # Scarico i dati dei prezzi di S&P500 da Yahoo Finance
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    sp500 = yf.download("^GSPC", start, end, group_by='ticker', auto_adjust=False)
    sp500 = sp500["^GSPC"]

    # Visualizzo il grafico a candele dei prezzi dell'ultimo anno
    plot_candlesticks(sp500, datetime.datetime(2015, 1, 1))

    # Eseguo il K-Means clustering con 5 clusters sui dati
    # 3-dimensionali di H/O, L/O e C/O
    sp500_norm = get_open_normalised_prices(sp500)
    k = 5
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(sp500_norm)
    labels = km.labels_
    sp500["Cluster"] = labels

    # Visualizzo il grafico 3D delle candele normalizzate usando H/O, L/O, C/O
    plot_3d_normalised_candles(sp500_norm, labels)

    # Visualizzo le candele OHLC riordinate
    # nei loro rispettivi cluster
    plot_cluster_ordered_candles(sp500)

    # Creo e restituisco la matrice dei cluster follow-on
    create_follow_cluster_matrix(sp500)
    print()