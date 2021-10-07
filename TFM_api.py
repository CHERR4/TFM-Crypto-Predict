import re
import joblib
import json
import pandas as pd
import numpy as np
from math import expm1
from flask import Flask, jsonify, request
from tensorflow import keras
from models.multivariable.VanillaMultivariableLSTM import VanillaLSTM
from models.multivariable.BidirectionalMultivariableLSTM import BidirectionalLSTM

app = Flask(__name__)
host = '0.0.0.0'
port = 3000

model_3_name = 'trainedModels/predict3/bidirectionalLSTM.h5'
model_3_params = 'trainedModels/predict3/paramsBidirectionalLSTM.json'
model_3_scaler = 'trainedModels/predict3/scalerBidirectionalLSTM.pkl'

model_2_name = 'trainedModels/predict2/bidirectionalLSTM.h5'
model_2_params = 'trainedModels/predict2/paramsBidirectionalLSTM.json'
model_2_scaler = 'trainedModels/predict2/scalerBidirectionalLSTM.pkl'

model_1_name = 'trainedModels/predict1/bidirectionalLSTM.h5'
model_1_params = 'trainedModels/predict1/paramsBidirectionalLSTM.json'
model_1_scaler = 'trainedModels/predict1/scalerBidirectionalLSTM.pkl'

model_predict_3 = BidirectionalLSTM()
model_predict_3.import_model(model_3_name, model_3_params, model_3_scaler)

model_predict_2 = BidirectionalLSTM()
model_predict_2.import_model(model_2_name, model_2_params, model_2_scaler)

model_predict_1 = BidirectionalLSTM()
model_predict_1.import_model(model_1_name, model_1_params, model_1_scaler)

def build_prediction_dataframe(training_ts, predictions_ts):
  last_training_predictions = [training_ts.tail(1), predictions_ts]
  last_training_predictions_ts = pd.concat(last_training_predictions)
  last_training_predictions_df = last_training_predictions_ts.reset_index()
  last_training_predictions_df['positive_delta'] = last_training_predictions_df.apply(lambda row: row.name > 0 and last_training_predictions_df.loc[row.name-1, :][last_training_predictions_df.columns[1]] < row[last_training_predictions_df.columns[1]], axis=1)
  predictions_df = last_training_predictions_df.iloc[1: , :][['positive_delta']]
  return predictions_df

@app.route('/')
def index():
    return 'App running'

@app.route('/predict-3', methods=['POST'])
def predict_3():
    n_features = 3
    n_steps = 7
    n_predict = 3
    steps_param = request.args.get('steps')
    steps_to_predict = [float(x) for x in steps_param.replace('[', '').replace(']','').split(',')]
    steps_with_zeros = steps_to_predict[:]

    steps_to_predict = np.reshape(steps_to_predict, (n_steps, n_features))
    steps_to_predict_df = pd.DataFrame(steps_to_predict)

    steps_with_zeros.extend(np.zeros(n_features*n_predict))
    steps_with_zeros = np.reshape(steps_with_zeros, (n_steps+n_predict, n_features))
    df = pd.DataFrame(steps_with_zeros)
    predictions = model_predict_3.predict(df)
    predictions = predictions.reshape(predictions.shape[0])
    predictions_ts = pd.DataFrame({df.columns[0]: predictions}, df.index[-n_predict:])
    predictions_df = build_prediction_dataframe(steps_to_predict_df, predictions_ts)
    predictions_df.rename(columns={'positive_delta': 'positive_delta_predict'}, inplace=True)
    output = {
        'prices': str(predictions),
        'price_go_up': str(np.reshape(predictions_df.values, predictions_df.values.shape[0]))
    }
    return jsonify(output)

@app.route('/retrain-3', methods=['POST'])
def retrain_3():
    n_features = 3
    n_steps = 7
    n_predict = 3
    stocks_param = request.args.get('stocks')
    stocks_to_train = [float(x) for x in stocks_param.replace('[', '').replace(']','').split(',')]
    stocks_formated = np.reshape(stocks_to_train, (int(len(stocks_to_train)/n_features), n_features))
    df = pd.DataFrame(stocks_formated)
    model_predict_3.retrain(df, epochs=1, batch_size=len(df), validation_split=0)
    return 'Model retrained'

@app.route('/info-model-3', methods=['GET'])
def info_3():
    with open(model_3_params) as params_file:
      params = json.load(params_file)
    return jsonify(params)

@app.route('/predict-2', methods=['POST'])
def predict_2():
    n_features = 3
    n_steps = 3
    n_predict = 2
    steps_param = request.args.get('steps')
    steps_to_predict = [float(x) for x in steps_param.replace('[', '').replace(']','').split(',')]
    steps_with_zeros = steps_to_predict[:]

    steps_to_predict = np.reshape(steps_to_predict, (n_steps, n_features))
    steps_to_predict_df = pd.DataFrame(steps_to_predict)

    steps_with_zeros.extend(np.zeros(n_features*n_predict))
    steps_with_zeros = np.reshape(steps_with_zeros, (n_steps+n_predict, n_features))
    df = pd.DataFrame(steps_with_zeros)
    predictions = model_predict_2.predict(df)
    predictions = predictions.reshape(predictions.shape[0])
    predictions_ts = pd.DataFrame({df.columns[0]: predictions}, df.index[-n_predict:])
    predictions_df = build_prediction_dataframe(steps_to_predict_df, predictions_ts)
    predictions_df.rename(columns={'positive_delta': 'positive_delta_predict'}, inplace=True)
    output = {
        'prices': str(predictions),
        'price_go_up': str(np.reshape(predictions_df.values, predictions_df.values.shape[0]))
    }
    return jsonify(output)

@app.route('/retrain-2', methods=['POST'])
def retrain_2():
    n_features = 3
    n_steps = 3
    n_predict = 2
    stocks_param = request.args.get('stocks')
    stocks_to_train = [float(x) for x in stocks_param.replace('[', '').replace(']','').split(',')]
    stocks_formated = np.reshape(stocks_to_train, (int(len(stocks_to_train)/n_features), n_features))
    df = pd.DataFrame(stocks_formated)
    model_predict_2.retrain(df, epochs=1, batch_size=len(df), validation_split=0)
    return 'Model retrained'

@app.route('/info-model-2', methods=['GET'])
def info_2():
    with open(model_2_params) as params_file:
      params = json.load(params_file)
    return jsonify(params)

@app.route('/predict-1', methods=['POST'])
def predict_1():
    n_features = 3
    n_steps = 5
    n_predict = 1
    steps_param = request.args.get('steps')
    steps_to_predict = [float(x) for x in steps_param.replace('[', '').replace(']','').split(',')]
    steps_with_zeros = steps_to_predict[:]

    steps_to_predict = np.reshape(steps_to_predict, (n_steps, n_features))
    steps_to_predict_df = pd.DataFrame(steps_to_predict)

    steps_with_zeros.extend(np.zeros(n_features*n_predict))
    steps_with_zeros = np.reshape(steps_with_zeros, (n_steps+n_predict, n_features))
    df = pd.DataFrame(steps_with_zeros)
    predictions = model_predict_1.predict(df)
    predictions = predictions.reshape(predictions.shape[0])
    predictions_ts = pd.DataFrame({df.columns[0]: predictions}, df.index[-n_predict:])
    predictions_df = build_prediction_dataframe(steps_to_predict_df, predictions_ts)
    predictions_df.rename(columns={'positive_delta': 'positive_delta_predict'}, inplace=True)
    output = {
        'prices': str(predictions),
        'price_go_up': str(np.reshape(predictions_df.values, predictions_df.values.shape[0]))
    }
    return jsonify(output)

@app.route('/retrain-1', methods=['POST'])
def retrain_1():
    n_features = 3
    n_steps = 5
    n_predict = 1
    stocks_param = request.args.get('stocks')
    stocks_to_train = [float(x) for x in stocks_param.replace('[', '').replace(']','').split(',')]
    stocks_formated = np.reshape(stocks_to_train, (int(len(stocks_to_train)/n_features), n_features))
    df = pd.DataFrame(stocks_formated)
    model_predict_1.retrain(df, epochs=1, batch_size=len(df), validation_split=0)
    return 'Model retrained'

@app.route('/info-model-1', methods=['GET'])
def info_1():
    with open(model_1_params) as params_file:
      params = json.load(params_file)
    return jsonify(params)

if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)
    print('Api running on host:', host, ', port:', port)