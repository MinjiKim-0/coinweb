import json
import requests
import pyupbit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
# from keras.models import load_model
# import tensorflow as tf

scaler = MinMaxScaler()


def get_train_price_data(ticker):
    df = pyupbit.get_ohlcv(ticker, count=66)
    df.drop_duplicates(inplace=True)
    array = df.to_numpy()
    return array


def scaling(array):
    array_scaled = scaler.fit_transform(array)
    return array_scaled


def create_dataset(dataset, look_back=7, foresight=(30-1)):
    X = []
    for i in range(dataset.shape[0]-look_back - foresight):
        obs = dataset[i:(i+look_back), :]
        X.append(obs)
    return np.array(X)


def getJSONdata(ticker, x_pred):
    ticker_name = ticker.lstrip("KRW-")

    input_data_jason = json.dumps({
        "signiture_name": f"serving_{ticker_name}_lstm",
        "instances": x_pred.tolist()
    })
    return input_data_jason


# SERVER_URL_BTC = 'http://localhost:8501/v1/models/btc_lstm:predict'
# SERVER_URL_ETH = 'http://localhost:8502/v1/models/eth_lstm:predict'  # 127.0.0.1


def getResponse(ticker, input_data_jason):
    ticker_name = ticker.lstrip("KRW-")
    ticker_low = ticker_name.lower()
    if ticker_low == 'btc':
        portnumber = '1'
    else:
        portnumber = '2'

    response = requests.post(
        f'http://localhost:850{portnumber}/v1/models/{ticker_low}_lstm:predict', data=input_data_jason)
    response.raise_for_status()
    response = response.json()
    return response


def getPreds(response):
    preds = np.array(response["predictions"])
    preds = np.concatenate([preds, preds, preds, preds, preds, preds], axis=1)
    preds_inversed = scaler.inverse_transform(preds)
    return preds_inversed[:, 0]


def predict(ticker):
    array = get_train_price_data(ticker)
    array_scaled = scaling(array)
    x_pred = create_dataset(array_scaled)
    x_pred = np.reshape(
        x_pred, (x_pred.shape[0], x_pred.shape[2], x_pred.shape[1]))
    # model = load_model('6_best_btc_30d_lb7_val_loss_0.0292.hdf5')
    # model = tf.keras.models.load_model(
    #     '10_latest_btc_30d_lb7_loss_0.0395_val_loss_0.0422.hdf5')
    input_data_jason = getJSONdata(ticker, x_pred)
    response = getResponse(ticker, input_data_jason)
    # preds_inversed = getPreds(response)
    preds = np.array(response["predictions"])
    preds = np.concatenate([preds, preds, preds, preds, preds, preds], axis=1)
    preds_inversed = scaler.inverse_transform(preds)
    preds_inversed = preds_inversed[:, 0]

    return preds_inversed


def getLatest(ticker):
    price_latest = pyupbit.get_current_price(ticker)
    return price_latest


def getMonthdays():
    today = pd.Timestamp.today().strftime('%m-%d-%y')
    mdays = pd.date_range(today, periods=31).strftime('%y-%m-%d')[1:]
    mdays = list(mdays)
    return mdays
