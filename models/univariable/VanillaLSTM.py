import json
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

class VanillaLSTM:

  def __init__(self, n_neurons=None, n_steps=1, n_features=1, n_outputs=3, loss='mean_squared_error', optimizer='adam'):
    if n_neurons is not None:
      self.model = Sequential()
      self.model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
      self.model.add(Dense(n_outputs))
      self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
      self.n_steps = n_steps
      self.n_features = n_features
      self.n_outputs = n_outputs
      self.loss = loss
      self.optimizer = optimizer
      self.scaler = MinMaxScaler(feature_range=(0, 1)) # Default scaler, not other scaler used yet
      print(self.model.summary())
    else:
      print('Empty model remember to import an existing model')


  def __split_sequence(self, sequence):
    X, y = list(), list()
    if self.n_steps == len(sequence):
      X.append(sequence)
    else:
      for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + self.n_steps + self.n_outputs - 1
        train_ix = i + self.n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
          break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:train_ix], sequence[train_ix:end_ix+1]

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

  def train(self, train_df, epochs, batch_size=16, validation_split=0.1):
    # train_df.reset_index(inplace=True)
    train_scaled = self.scaler.fit_transform(train_df)
    train_x, train_y = self.__split_sequence(train_scaled)
    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], self.n_steps, self.n_features))

    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))

    history = self.model.fit(train_x, train_y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
    return history.history

  def retrain(self, train_df, epochs, batch_size=16, validation_split=0.1):
    # train_df.reset_index(inplace=True)
    train_scaled = self.scaler.transform(train_df)
    train_x, train_y = self.__split_sequence(train_scaled)
    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], self.n_steps, self.n_features))

    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))

    history = self.model.fit(train_x, train_y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
    return history.history
  
  def predict(self, test_df):
    test_scaled = self.scaler.transform(test_df)
    test_x, _ = self.__split_sequence(test_scaled)
    test_x = np.reshape(test_x, (test_x.shape[0], self.n_steps, self.n_features))
    predictions = self.model.predict(test_x)
    predictions = self.scaler.inverse_transform(predictions)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    return predictions

  def export_model(self, model_path='vanillaLSTM.h5', params_path='paramsVanillaLSTM.json', scaler_path='scalerVanillaLSTM.pkl'):
    self.model.save(model_path)
    params = {
      'n_steps': self.n_steps,
      'n_features': self.n_features,
      'n_outputs': self.n_outputs,
      'loss': self.loss,
      'optimizer': self.optimizer
    }
    with open(params_path, 'w') as params_file:
      json.dump(params, params_file,  indent=4)
    dump(self.scaler, open(scaler_path, 'wb'))

  def import_model(self, model_path='vanillaLSTM.h5', params_path='paramsVanillaLSTM.json', scaler_path='scalerVanillaLSTM.pkl'):
    self.model = load_model(model_path)
    with open(params_path) as params_file:
      params = json.load(params_file)
    self.n_steps = params['n_steps']
    self.n_features = params['n_features']
    self.n_outputs = params['n_outputs']
    self.loss = params['loss']
    self.optimizer = params['optimizer']
    self.scaler = load(open(scaler_path, 'rb'))
