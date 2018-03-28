from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import tensorflow as tf  # uncomment this for using GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# maximun alloc gpu50% of MEM
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# allocate dynamically
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import math
import json
import os
import sys

import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, ZeroPadding1D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd
import dataset
import argparse

import time
from datetime import timedelta


def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count * 4 // 5
    print("train size : {}".format(train_size))
    X_train = X[:train_size]
    y_train = y[:train_size]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_test = X[train_size:]
    y_test = y[train_size:]
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test, nb_classes


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # The identity_block is the block that has no conv layer at shortcut

    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    # conv_block is the block that has a conv layer at shortcut

    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    shortcut = BatchNormalization(name=bn_name_base + '1')(input_tensor)
    shortcut = Conv1D(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_model(SHAPE, nb_classes, seed=None):
    # [3, 24, 36, 3] resnet 200 architecture
    # error.
    if seed:
        np.random.seed(seed)

    input_tensor = Input(shape=(SHAPE))
    print(input_tensor)

    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    for i in range(0, 2):
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b{}'.format(i))
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(0, 7):
        x = identity_block(x, 3, [128, 128, 512],
                           stage=3, block='c{}'.format(i))
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(0, 35):
        x = identity_block(x, 3, [256, 256, 1024],
                           stage=4, block='d{}'.format(i))
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    for i in range(0, 2):
        x = identity_block(x, 3, [512, 512, 2048],
                           stage=5, block='e{}'.format(i))
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # print(x)
    x = AveragePooling1D(1, name='avg_pool')(x)

    x = Flatten()(x)

    model = Model(input_tensor, x)

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-i', '--input',
    #                     help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=150)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-o', '--optimizer',
                        help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    # SHAPE = (img_width, img_height ,channel)
    # Handle Dimension Ordering for different backends
    img_input=(120,4)

    # data_directory = args.input
    # period_name = data_directory.split('/')

    # load dataset
    dataframe = pd.read_csv("iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]
    from sklearn.preprocessing import LabelEncoder
    encoder =  LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    from sklearn.model_selection import train_test_split
    X_train,X_test, Y_train,y_test = train_test_split(X,dummy_y,test_size=0.2,random_state=0)
    model = build_model(img_input, 3)
    print(X_train.shape)
    model.compile(optimizer=args.optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Save Model or creates a HDF5 file
    # model.save('{}epochs_{}period_{}dimension_resnet152_{}.h5'.format(
    #     epochs, period_name[1], args.dimension, args.optimizer), overwrite=True)
    # del model  # deletes the existing model
    train_score = model.evaluate(X_train, Y_train, verbose=1)
    print('Overall Train score: {}'.format(train_score[0]))
    print('Overall Train accuracy: {}'.format(train_score[1]))

    test_score = model.evaluate(X_test, Y_test, verbose=1)
    print('Overall Test score: {}'.format(test_score[0]))
    print('Overall Test accuracy: {}'.format(test_score[1]))
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
