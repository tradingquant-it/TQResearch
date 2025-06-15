from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    # Set up the X and Y dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # CORRETTO: uso moderno per 3D

    X = np.arange(0, 20, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)

    # Coefficienti della normale univariata condizionata
    beta0 = -5.0
    beta1 = 0.5
    Z = norm.pdf(Y, beta0 + beta1 * X, 1.0)

    # Traccia la superficie con la mappa "coolwarm"
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False
    )

    # Imposta i limiti e la formattazione dell'asse z
    ax.set_zlim(0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Etichette degli assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P(Y|X)')

    # Angolo di visualizzazione
    ax.view_init(elev=30., azim=50.0)

    # Mostra il grafico
    plt.show()
