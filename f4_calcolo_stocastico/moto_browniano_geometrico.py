import os
import random
import string

import click
import numpy as np
import pandas as pd


class GeometricBrownianMotionAssetSimulator:
    """
    Questa classe richiamabile genererà un DataFrame dei prezzi giornalieri di
    apertura-massimo-minimo-chiusura-volumi (OHLCV) per simulare i percorsi di
    prezzo delle azioni con il moto browniano geometrico per il prezzo e una
    distribuzione di Pareto per il volume.

    Produrrà i risultati in un CSV con un simbolo ticker generato casualmente.

    Per ora lo strumento è hardcoded per generare dati giornalieri dei
    giorni lavorativo tra due date, incluse.

    Si noti che i dati sui prezzi e sul volume sono completamente non correlati,
    il che non è probabile che si verifichi per i dati di asset reali.

    Parameters
    ----------
    start_date : `str`
        La data di inizio nel formato AAAA-MM-GG.
    end_date : `str`
        La data di fine nel formato AAAA-MM-GG.
    output_dir : `str`
         Il percorso completo della directory di output per il file CSV.
    symbol_length : `int`
        La lunghezza da usare per il simbolo ticker.
    init_price : `float`
        Il prezzo iniziale dell'asset.
    mu : `float`
        La "deriva" media dell'asset.
    sigma : `float`
        La "volatilità" dell'asset.
    pareto_shape : `float`
        Il parametro utilizzato per governare la forma di distribuzione
        di Pareto per la generazione dei dati di volume.
    """

    def __init__(
        self,
        start_date,
        end_date,
        output_dir,
        symbol_length,
        init_price,
        mu,
        sigma,
        pareto_shape
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.symbol_length = symbol_length
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.pareto_shape = pareto_shape

    def _generate_random_symbol(self):
        """
        Genera una stringa di simbolo ticker casuale composta da caratteri
        ASCII maiuscoli da utilizzare nel nome file di output CSV.

        Returns
        -------
        `str`
            La stringa ticker casuale composta da lettere maiuscole.
        """
        return ''.join(
            random.choices(
                string.ascii_uppercase,
                k=self.symbol_length
            )
        )

    def _create_empty_frame(self):
        """
        Crea il DataFrame Pandas vuoto con una colonna date
        utilizzando i giorni lavorativi tra due date. Ognuna
        delle colonne prezzo/volume è impostata su zero.

        Returns
        -------
        `pd.DataFrame`
            DataFrame OHLCV vuoto per il popolamento successivo.
        """
        date_range = pd.date_range(
            self.start_date,
            self.end_date,
            freq='B'
        )

        zeros = pd.Series(np.zeros(len(date_range)))

        return pd.DataFrame(
            {
                'date': date_range,
                'open': zeros,
                'high': zeros,
                'low': zeros,
                'close': zeros,
                'volume': zeros
            }
        )[['date', 'open', 'high', 'low', 'close', 'volume']]

    def _create_geometric_brownian_motion(self, data):
        """
        Calcola il percorso del prezzo di un asset utilizzando la
        soluzione analitica dell'equazione differenziale stocastica
        (SDE) del moto browniano geometrico.

        Questo divide il solito timestep per quattro in modo che la
        serie dei prezzi sia quattro volte più lunga, per tenere conto
        della necessità di avere un prezzo di apertura, massimo, minimo
         e chiusura per ogni giorno. Questi prezzi vengono successivamente
        delimitati correttamente in un ulteriore metodo.

        Parameters
        ----------
        data : `pd.DataFrame`
            Il DataFrame necessario per calcolare la lunghezza delle serie temporali.

        Returns
        -------
        `np.ndarray`
            Il percorso del prezzo dell'asset (quattro volte più lungo per includere OHLC).
        """
        n = len(data)
        T = n / 252.0  # Giorni lavorativi in un anno
        dt = T / (4.0 * n)  # 4.0 è necessario perchè sono richiesti quattro prezzi per ogni giorno

        # implementazione vettorializzata per la generazione di percorsi di asset
        # includendo quattro prezzo per ogni giorni, usati per creare OHLC
        asset_path = np.exp(
            (self.mu - self.sigma ** 2 / 2) * dt +
            self.sigma * np.random.normal(0, np.sqrt(dt), size=(4 * n))
        )

        return self.init_price * asset_path.cumprod()

    def _append_path_to_data(self, data, path):
        """
        Tiene conto correttamente dei calcoli massimo/minimo necessari
        per generare un prezzo massimo e minimo corretto per il
        prezzo di un determinato giorno.

        Il prezzo di apertura prende ogni quarto valore, mentre il
        prezzo di chiusura prende ogni quarto valore sfalsato di 3
        (ultimo valore in ogni blocco di quattro).

        I prezzi massimo e minimo vengono calcolati prendendo il massimo
        (risp. minimo) di tutti e quattro i prezzi in un giorno e
        quindi aggiustando questi valori se necessario.

        Tutto questo viene eseguito sul posto in modo che il frame
        non venga restituito tramite il metodo.


        Parameters
        ----------
        data : `pd.DataFrame`
            Il DataFrame prezzo/volume da modificare sul posto.
        path : `np.ndarray`
            L'array NumPy originale del percorso del prezzo dell'asset.
        """
        data['open'] = path[0::4]
        data['close'] = path[3::4]

        data['high'] = np.maximum(
            np.maximum(path[0::4], path[1::4]),
            np.maximum(path[2::4], path[3::4])
        )

        data['low'] = np.minimum(
            np.minimum(path[0::4], path[1::4]),
            np.minimum(path[2::4], path[3::4])
        )

    def _append_volume_to_data(self, data):
        """
        Utilizza una distribuzione di Pareto per simulare dati di volume
        non negativi. Si noti che questo non è correlato al prezzo
        dell'attività sottostante, come sarebbe probabilmente il caso dei
        dati reali, ma è un'approssimazione ragionevolmente efficace.

        Parameters
        ----------
        data : `pd.DataFrame`
            Il DataFrame a cui aggiungere i dati del volume, sul posto.
        """
        data['volume'] = np.array(
            list(
                map(
                    int,
                    np.random.pareto(
                        self.pareto_shape,
                        size=len(data)
                    ) * 1000000.0
                )
            )
        )

    def _output_frame_to_dir(self, symbol, data):
        """
        Output the fully-populated DataFrame to disk into the
        desired output directory, ensuring to trim all pricing
        values to two decimal places.
        Memorizza il DataFrame completamente popolato su disco
        nella directory di output desiderata, assicurandosi di
        ridurre tutti i valori dei prezzi a due cifre decimali.

        Parameters
        ----------
        symbol : `str`
            Il simbolo ticker con cui denominare il file.
        data : `pd.DataFrame`
            DataFrame contenente i dati OHLCV generati.
        """
        output_file = os.path.join(self.output_dir, '%s.csv' % symbol)
        data.to_csv(output_file, index=False, float_format='%.2f')

    def __call__(self):
        """
        Il punto di ingresso per la generazione del frame OHLCV dell'asset. Si genera
        un simbolo e un dataframe vuoto. Quindi popola questo dataframe con alcuni
        dati GBM simulati. Il volume dell'asset viene quindi aggiunto a questi
        dati e infine viene salvato su disco come CSV.
        """
        symbol = self._generate_random_symbol()
        data = self._create_empty_frame()
        path = self._create_geometric_brownian_motion(data)
        self._append_path_to_data(data, path)
        self._append_volume_to_data(data)
        self._output_frame_to_dir(symbol, data)


@click.command()
@click.option('--num-assets', 'num_assets', default='1', help='Numero di asset separati per cui generare file')
@click.option('--random-seed', 'random_seed', default='42', help='Seed casuale da impostare sia per Python che per NumPy per la riproducibilità')
@click.option('--start-date', 'start_date', default=None, help='La data di inizio per la generazione dei dati sintetici nel formato AAAA-MM-GG')
@click.option('--end-date', 'end_date', default=None, help='La data di inizio per la generazione dei dati sintetici nel formato AAAA-MM-GG')
@click.option('--output-dir', 'output_dir', default=None, help='La posizione in cui inviare il file CSV di dati sintetici')
@click.option('--symbol-length', 'symbol_length', default='5', help='La lunghezza del simbolo dell''asset utilizzando caratteri ASCII maiuscoli')
@click.option('--init-price', 'init_price', default='100.0', help='Il prezzo iniziale da utilizzare')
@click.option('--mu', 'mu', default='0.1', help='Il parametro di deriva, \mu per GBM SDE')
@click.option('--sigma', 'sigma', default='0.3', help='Il parametro di volatilità, \sigma per SDE GBM')
@click.option('--pareto-shape', 'pareto_shape', default='1.5', help='La forma della distribuzione di Pareto che simula il volume degli scambi')
def cli(num_assets, random_seed, start_date, end_date, output_dir, symbol_length, init_price, mu, sigma, pareto_shape):
    num_assets = int(num_assets)
    random_seed = int(random_seed)
    symbol_length = int(symbol_length)
    init_price = float(init_price)
    mu = float(mu)
    sigma = float(sigma)
    pareto_shape = float(pareto_shape)

    # Seed per Python e NumPy
    random.seed(random_seed)
    np.random.seed(seed=random_seed)

    gbmas = GeometricBrownianMotionAssetSimulator(
        start_date,
        end_date,
        output_dir,
        symbol_length,
        init_price,
        mu,
        sigma,
        pareto_shape
    )

    # Crea num_assets file tramite la chiamata
    # ripetuta alla classe istanziata
    for i in range(num_assets):
        print('Generating asset path %d of %d...' % (i+1, num_assets))
        gbmas()


if __name__ == "__main__":
    cli()