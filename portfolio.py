import numpy as np
import pyupbit
from sklearn.preprocessing import MinMaxScaler
import random
from predict import getJSONdata, getResponse, getPreds


def get_train_price_data(ticker):
    df = pyupbit.get_ohlcv(ticker, count=1500)
    return df


def getArrayindex(ticker):
    df = get_train_price_data(ticker)
    df.drop_duplicates(inplace=True)
    array = df.to_numpy()
    idx = df.index
    dates = []
    for i in range(len(idx)):
        d = idx[i].strftime('%y-%m-%d')
        dates.append(d)
    return array, dates


def create_dataset(dataset, look_back=7, foresight=(30-1)):
    X, Y = [], []

    for i in range(dataset.shape[0]-look_back - foresight):
        obs = dataset[i:(i+look_back), :]
        X.append(obs)
        Y.append(dataset[i+(look_back+foresight), 0])

    return np.array(X), np.array(Y)


def cutFourFifth(data):
    fourfifth = len(data)-int(len(data)/5)
    return data[fourfifth:]


def getTPA(ticker):  # , model
    array, dates = getArrayindex(ticker)
    test_dates = cutFourFifth(dates)
    test_data = cutFourFifth(array)

    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(test_data)
    x_test, y_test = create_dataset(test_data)
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1]))

    input_data_jason = getJSONdata(ticker, x_test)
    response = getResponse(ticker, input_data_jason)
    # preds_inversed = getPreds(response)
    preds = np.array(response["predictions"])
    preds = np.concatenate([preds, preds, preds, preds, preds, preds], axis=1)
    preds_inversed = scaler.inverse_transform(preds)
    preds_inversed = preds_inversed[:, 0]

    # preds = model.predict(x_test)
    # preds = np.concatenate([preds, preds, preds, preds, preds, preds], axis=1)
    # preds_inversed = scaler.inverse_transform(preds)

    y_test = y_test.reshape(-1, 1)
    y_test = np.concatenate(
        [y_test, y_test, y_test, y_test, y_test, y_test], axis=1)
    y_test_inversed = scaler.inverse_transform(y_test)

    box = list(range(30, preds_inversed.shape[0]))
    a = random.sample(box, 5)
    a = sorted(a)
    aa = reversed(a)
    a = list(aa)

    term = []
    predicted = []
    actual = []
    for i in a:
        d = test_dates[i-30:i]
        term.append(d)

        p = preds_inversed[i-30:i]
        p = p.tolist()
        predicted.append(p)

        y = y_test_inversed[i-30:i, 0]
        y = y.tolist()
        actual.append(y)

    return term, predicted, actual
