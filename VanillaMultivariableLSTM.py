import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

class VanillaLSTM:

  def __init__(self, n_neurons=50, n_steps=1, n_features=1, n_outputs=3, loss='mean_squared_error', optimizer='adam'):
    self.model = Sequential()
    self.model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
    self.model.add(Dense(n_outputs))
    self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    self.n_steps = n_steps
    self.n_features = n_features
    self.n_outputs = n_outputs
    self.scaler = MinMaxScaler(feature_range=(0, 1))
    print(self.model.summary())


  def series_to_supervised(self, data, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(self.n_steps, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, self.n_outputs):
      cols.append(df.shift(-i))
      if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)
    return agg

  def train(self, train_df, epochs, batch_size=16, validation_split=0.1):
    # train_df.reset_index(inplace=True)
    values = train_df.values
    values = values.astype('float32')

    scaled = self.scaler.fit_transform(values)

    train_formated_df = self.series_to_supervised(scaled)
    train = train_formated_df.values
    # reshape input to be [samples, time steps, features]
    n_obs = self.n_steps * self.n_features
    train_X, train_y = train[:, :n_obs], train[:, -self.n_features]
    train_X = train_X.reshape((train_X.shape[0], self.n_steps, self.n_features))
    history = self.model.fit(train_X, train_y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
    return history.history
  
  def predict(self, test_df):
    values = test_df.values
    values = values.astype('float32')
    scaled = self.scaler.transform(values)
    print('scaled\n', scaled)
    test_formated_df = self.series_to_supervised(scaled)
    print("test_formated_df", test_formated_df.head())
    test = test_formated_df.values
    n_obs = self.n_steps * self.n_features
    test_X, test_y = test[:, :n_obs], test[:, -self.n_features]
    test_x = np.reshape(test_X, (test_X.shape[0], self.n_steps, self.n_features))
    predictions = self.model.predict(test_x)
    print("predictions", predictions)
    print(predictions.shape)
    last_two_columns = np.array([x[-2:] for x in values])
    last_rows = last_two_columns[-self.n_outputs:]
    print("last rows", last_rows)
    print(last_rows.shape)
    y_concat = np.concatenate((predictions.T, last_rows), axis=1)
    print("concat", y_concat)
    predictions = self.scaler.inverse_transform(y_concat)
    print("predictions", predictions)
    predictions = np.array([x[:1] for x in predictions])
    # predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    return predictions

  def save_model(self, path='vanillaLSTM.h5'):
    self.model.save(path)

  def load_model(self, path='vanillaLSTM.h5'):
    self.model = load_model(path)

