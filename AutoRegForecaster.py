from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

class AutoRegForecaster:

    def __init__(self):
        self.model = forecaster_rf = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=0),
                    lags=15)
        self.seed = 0


    def set_seed(self, seed):
        self.seed = seed

        self.model = forecaster_rf = ForecasterAutoreg(
            regressor = RandomForestRegressor(random_state=seed),
            lags=15)

    def train(self, train_df):
        self.model.fit(y=train_df.close)
    
    def predict(self, n_predicts):
        predictions = self.model.predict(steps= n_predicts)
        return predictions