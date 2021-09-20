# TFM-Crypto-Predict-LSTM
Autor: Jose Ramón Casero Fuentes

Este repositorio tiene la siguiente estructura:
    - En el repositorio raíz podemos ver los cuadernos utilizados para las etapas
    de preparación de los datos y posteriormente, la búsqueda de los hiperparámetros
    y prueba de los modelos
    - `data`: carpeta con todos los archivos de datos del proyecto, necesarios
    para las diferentes etapas. Origen `TFM_Data_Preparation.ipynb`.
    - `models`: dispone de todos los modelos desarrollados durante el proyecto.
    Tiene dos carpetas para distinguir entre modelos que funcionan con una
    única variable, y modelos que funcionan con múltiples variables.

  1. **Preparación de los datos**
  - `TFM_Data_Preparation.ipynb`
  - salida:
    - `stocks.csv`
    - `btcStocks.csv`
    - `ethStocks.csv`
    - `adaStocks.csv`
    - `btcActualStocks.csv`
    - `ethActualStocks.csv`
    - `adaActualStocks.csv`
    
    
  2. **Búsqueda hiperparámetros**
    2.1. **Univariable**
    - `TFM_Monovariable_Btc_Forecast.ipynb`
    - Ficheros de entrada para su ejecución:
        - univariable/btcStocks.csv
        - models/univariable/*.py
    2.2. **Multivariable**
    - `TFM_Multivariable_Forecast.ipynb`
        - multivariable/stocks.csv
        - models/multivariable/*.py

  3. **Prueba distintas series**
    - `TFM_Different_Stocks_Test.ipynb`
    - Ficheros de entrada para su ejecución:
        - multivariable/stocks.csv
        - models/univariable/*.py

  4. **Pruebas datos futuros**
    - `TFM_New_Dates_Predictions.ipynb`
    - Ficheros de entrada para su ejecución:
        - univariable/*.csv
        - actualUnivariable/*.csv
        - models/univariable/*.py