# Import the Time Series library
import statsmodels.tsa.stattools as ts

# Import Datetime and the Pandas DataReader
import yfinance as yf
from datetime import datetime

start_date = datetime(2000, 1, 1)
end_date = datetime(2013, 1, 1)

df = yf.download("GOOG", start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
goog = df['GOOG']

# Esegui il test ADF
result = ts.adfuller(goog['Adj Close'], maxlag=1)

# Visualizza i risultati
print('Statistiche ADF:', result[0])
print('p-value:', result[1])
print('Lags usati:', result[2])
print('Numero di osservazioni:', result[3])
print('Valori critici:')
for key, value in result[4].items():
    print(f'   {key}: {value}')