import keras

import keras_resnet.models

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets


def get_stock_data(normalize=True):
    df = pd.read_csv('SPY.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(df['date'], inplace=True)
    del df['date']
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['adjOpen'] = min_max_scaler.fit_transform(
            df.adjOpen.values.reshape(-1, 1))
        df['adjHigh'] = min_max_scaler.fit_transform(
            df.adjHigh.values.reshape(-1, 1))
        df['adjLow'] = min_max_scaler.fit_transform(
            df.adjLow.values.reshape(-1, 1))
        df['adjVolume'] = min_max_scaler.fit_transform(
            df.adjVolume.values.reshape(-1, 1))
        df['adjClose'] = min_max_scaler.fit_transform(
            df.adjClose.values.reshape(-1, 1))
    return df


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    # maxmimum date = lastest date - sequence length
    for index in range(len(data) - sequence_length):
        # index : index + 22days
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(
        X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]

seq_len = 22

shape, classes = (1, 4), 1

x = keras.layers.Input(shape)

model =keras_resnet.models.ResNet18_1d(x, classes=classes, include_top=False)

model.compile("adam", "categorical_crossentropy", ["accuracy"])

df = get_stock_data(normalize=True)

# training_x, training_y, X_test, y_test = load_data(df, seq_len)

iris = datasets.load_iris()
training_x = iris.data  # we only take the first two features.
training_y = iris.target
print("training_x : {}".format(training_y))
training_y = keras.utils.np_utils.to_categorical(training_y)
model.fit(training_x, training_y)
