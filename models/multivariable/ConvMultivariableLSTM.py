import json
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten, ConvLSTM2D
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load


class ConvLSTM:

  def __init__(self, n_neurons=None, n_steps=1, n_features=1, n_outputs=3, n_filters=64, kernel_size=(1,2), n_seq=1, loss='mean_squared_error', optimizer='adam'):
    if n_neurons is not None:
      self.model = Sequential()
      self.model.add(ConvLSTM2D(filters=n_filters, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
      self.model.add(Flatten())
      self.model.add(Dense(n_outputs))
      self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
      self.n_neurons = n_neurons
      self.n_steps = n_steps
      self.n_features = n_features
      self.n_outputs = n_outputs
      self.scaler = MinMaxScaler(feature_range=(0, 1))
      self.n_seq = n_seq
      self.loss = loss
      self.optimizer = optimizer
      print(self.model.summary())
    else:
      print('Empty model remember to import an existing model')



  # convert series to supervised learning
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
    # train_X = train_X.reshape((train_X.shape[0], self.n_steps, self.n_features))
    train_x = train_X.reshape((train_X.shape[0], self.n_seq, 1, self.n_steps, self.n_features))
    history = self.model.fit(train_x, train_y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
    return history.history

  def retrain(self, train_df, epochs, batch_size=16, validation_split=0.1):
    # train_df.reset_index(inplace=True)
    values = train_df.values
    values = values.astype('float32')

    scaled = self.scaler.transform(values)

    train_formated_df = self.series_to_supervised(scaled)
    train = train_formated_df.values
    # reshape input to be [samples, time steps, features]
    n_obs = self.n_steps * self.n_features
    train_X, train_y = train[:, :n_obs], train[:, -self.n_features]
    # train_X = train_X.reshape((train_X.shape[0], self.n_steps, self.n_features))
    train_x = train_X.reshape((train_X.shape[0], self.n_seq, 1, self.n_steps, self.n_features))
    history = self.model.fit(train_x, train_y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
    return history.history
  
  def predict(self, test_df):
    values = test_df.values
    values = values.astype('float32')
    scaled = self.scaler.transform(values)
    test_formated_df = self.series_to_supervised(scaled)
    test = test_formated_df.values
    n_obs = self.n_steps * self.n_features
    test_X, test_y = test[:, :n_obs], test[:, -self.n_features]
    # test_x = np.reshape(test_X, (test_X.shape[0], self.n_steps, self.n_features))
    test_x = test_X.reshape((test_X.shape[0], self.n_seq, 1, self.n_steps, self.n_features))

    predictions = self.model.predict(test_x)
    last_two_columns = np.array([x[-2:] for x in values])
    last_rows = last_two_columns[-self.n_outputs:]
    y_concat = np.concatenate((predictions.T, last_rows), axis=1)
    predictions = self.scaler.inverse_transform(y_concat)
    predictions = np.array([x[:1] for x in predictions])
    # predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    return predictions

  def export_model(self, model_path='convLSTM.h5', params_path='paramsConvLSTM.json', scaler_path='scalerConvLSTM.pkl'):
    self.model.save(model_path)
    params = {
      'model': 'ConvLSTM',
      'n_neurons': self.n_neurons,
      'n_steps': self.n_steps,
      'n_features': self.n_features,
      'n_outputs': self.n_outputs,
      'loss': self.loss,
      'optimizer': self.optimizer
    }
    with open(params_path, 'w') as params_file:
      json.dump(params, params_file,  indent=4)
    dump(self.scaler, open(scaler_path, 'wb'))

  def import_model(self, model_path='convLSTM.h5', params_path='paramsConvLSTM.json', scaler_path='scalerConvLSTM.pkl'):
    self.model = load_model(model_path)
    with open(params_path) as params_file:
      params = json.load(params_file)
    self.n_neurons = params['n_neurons']
    self.n_steps = params['n_steps']
    self.n_features = params['n_features']
    self.n_outputs = params['n_outputs']
    self.loss = params['loss']
    self.optimizer = params['optimizer']
    self.scaler = load(open(scaler_path, 'rb'))
    self.n_seq = params['n_seq']
