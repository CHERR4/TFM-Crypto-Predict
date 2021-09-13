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

  def save_model(self, path='vanillaLSTM.h5'):
    self.model.save(path)

  def load_model(self, path='vanillaLSTM.h5'):
    self.model = load_model(path)

