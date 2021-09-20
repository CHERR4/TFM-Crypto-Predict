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
    - `multivariable/stocks.csv`
    - `univariable/btcStocks.csv`
    - `univariable/ethStocks.csv`
    - `univariable/adaStocks.csv`
    - `actualUnivariable/btcActualStocks.csv`
    - `actualUnivariable/ethActualStocks.csv`
    - `actualUnivariable/adaActualStocks.csv`
    
    
  2. **Búsqueda hiperparámetros**<br />
    1. **Univariable**
    - `TFM_Monovariable_Btc_Forecast.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - univariable/btcStocks.csv
        - models/univariable/*.py
    2. **Multivariable**
    - `TFM_Multivariable_Forecast.ipynb`<br />
        - multivariable/stocks.csv
        - models/multivariable/*.py

  3. **Prueba distintas series**<br />
    - `TFM_Different_Stocks_Test.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - multivariable/stocks.csv
        - models/univariable/*.py

  4. **Pruebas datos futuros**<br />
    - `TFM_New_Dates_Predictions.ipynb`<br />
    - Ficheros de entrada para su ejecución:
        - univariable/*.csv
        - actualUnivariable/*.csv
        - models/univariable/*.py