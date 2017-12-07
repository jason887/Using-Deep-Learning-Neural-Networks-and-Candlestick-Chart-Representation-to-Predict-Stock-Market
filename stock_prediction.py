"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet
import pandas as pd
from sklearn import preprocessing


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
def dataPreprocessing(dataFile,normalize,seq_len):
    #data pre-processing
    data = pd.read_csv(dataFile,index_col=0)
    columnsTitles=["adjOpen","adjHigh","adjLow","adjVolume","adjClose"]
    data=data.reindex(columns=columnsTitles)
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        data['adjOpen'] = min_max_scaler.fit_transform(
            data.adjOpen.values.reshape(-1, 1))
        data['adjHigh'] = min_max_scaler.fit_transform(
            data.adjHigh.values.reshape(-1, 1))
        data['adjLow'] = min_max_scaler.fit_transform(
            data.adjLow.values.reshape(-1, 1))
        data['adjVolume'] = min_max_scaler.fit_transform(
            data.adjVolume.values.reshape(-1, 1))
        data['adjClose'] = min_max_scaler.fit_transform(
            data.adjClose.values.reshape(-1, 1))
    amount_of_features = len(data.columns)
    dataX = data.as_matrix()
    sequence_length = seq_len + 1
    result = []
    # maxmimum date = lastest date - sequence length
    for index in range(len(dataX) - sequence_length):
        # index : index + seq_len days
        result.append(dataX[index: index + sequence_length])
    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split
    train = result[:int(row), :]
    X_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]
#     X_train = X_train.reshape(len(X_train),2,2,1)
#     X_test = X_test.reshape(len(X_test),2,2,1)
    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1], amount_of_features,1))
    X_test = np.reshape(
        X_test, (X_test.shape[0], X_test.shape[1], amount_of_features,1))
    return [X_train, y_train, X_test, y_test]

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


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
    0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
# csv_logger = CSVLogger('resnet18_cifar10.csv')
csv_logger = CSVLogger('resnet_stockpredict.csv')

seq_len = 22
batch_size = 32
nb_classes = 1
nb_epoch = 5
data_augmentation = False

# input image dimensions
# img_rows, img_cols = 32, 32
img_rows = 1
img_cols = 5
# The CIFAR10 images are RGB.
# img_channels = 3
# stock data doesn't have channel
img_channels = 1

# The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = load_data('SPY.csv')
# prepare data
df = get_stock_data(normalize=True)
X_train, y_train, X_test, y_test = load_data(df, seq_len)

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.

model = resnet.ResnetBuilder.build_resnet_18(
    (img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, csv_logger])
