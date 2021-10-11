# TFM-Crypto-Predict-LSTM
**Autor:** Jose Ramón Casero Fuentes

Este repositorio tiene la siguiente estructura:
    <br />
    - En el repositorio raíz podemos ver los cuadernos utilizados para las etapas
    de preparación de los datos y posteriormente, la búsqueda de los hiperparámetros
    y prueba de los modelos.<br />
    - `data`: carpeta con todos los archivos de datos del proyecto, necesarios
    para las diferentes etapas. Origen `TFM_Data_Preparation.ipynb`.<br />
    - `models`: dispone de todos los modelos desarrollados durante el proyecto.
    Tiene dos carpetas para distinguir entre modelos que funcionan con una
    única variable, y modelos que funcionan con múltiples variables.<br />

  1. **Preparación de los datos**
  - `TFM_Data_Preparation.ipynb`
  - salida:
    - `data/multivariable/stocks.csv`
    - `data/univariable/btcStocks.csv`
    - `data/univariable/ethStocks.csv`
    - `data/univariable/adaStocks.csv`
    - `data/actualUnivariable/btcActualStocks.csv`
    - `data/actualUnivariable/ethActualStocks.csv`
    - `data/actualUnivariable/adaActualStocks.csv`
  - `TFM_Trends_Preparation.csv`
  - entrada:
    - `bitcoinWeekSearches.csv`
    - `bitcoinWeekNewsSearches.csv`
  - salida:
    - `stocksSearches.csv`
    
    
  2. **Búsqueda hiperparámetros**<br />
    1. **Univariable**
    - `TFM_Monovariable_Btc_Forecast.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - data/univariable/btcStocks.csv
        - models/univariable/*.py<br />
    2. **Multivariable**
    - `TFM_Multivariable_Forecast.ipynb`<br />
        - data/multivariable/stocks.csv
        - models/multivariable/*.py

  3. **Prueba distintas series**<br />
    - `TFM_Different_Stocks_Test.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - data/multivariable/stocks.csv
        - models/univariable/*.py


  4. **Pruebas datos futuros**<br />
    1. **Univariable**
    - `TFM_New_Dates_Predictions.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - data/univariable/*.csv
        - data/actualUnivariable/*.csv
        - models/univariable/*.py
    2. **Multivariable**
    - `TFM_New_Dates_Predictions_Multivariable.ipynb`<br/>
    - Ficheros de entrada para su ejecución:
        - data/multivariable/*.csv
        - models/multivariable/*.csv

