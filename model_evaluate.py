# import tensorflow as tf # uncomment this for using GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # comment this for using GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change with 1 for using GPU
# # uncomment below for using GPU
# config = tf.ConfigProto()
# # maximun alloc gpu50% of MEM
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# #allocate dynamically
# config.gpu_options.allow_growth = True
# sess = tf.Session(config = config)

import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.resnet50 import conv_block, identity_block
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, AveragePooling2D)
from keras import backend as K
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import dataset
import argparse

import time
from datetime import timedelta

from keras.models import load_model

def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count * 4 // 5
    print("train size : {}".format(train_size))
    X_train = X[:train_size]
    y_train = y[:train_size]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_test  = X[train_size:]
    y_test  = y[train_size:]
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test, nb_classes

def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=200)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-md', '--model_name',
                        help='a model name', type=str, required=True)
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="output.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    SHAPE = (img_width, img_height ,channel)
    bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

    data_directory = args.input
    period_name = data_directory.split('/')

    print ("loading dataset")
    X_train, Y_train, X_test, Y_test, nb_classes= build_dataset(data_directory, args.dimension)
    print("number of classes : {}".format(nb_classes))

    # load pre-trained model
    model = load_model(args.model_name)

    train_score = model.evaluate(X_train, Y_train, verbose=1)
    print('Overall Train score: {}'.format(train_score[0]))
    print('Overall Train accuracy: {}'.format(train_score[1]))

    test_score = model.evaluate(X_test, Y_test, verbose=1)
    print('Overall Test score: {}'.format(test_score[0]))
    print('Overall Test accuracy: {}'.format(test_score[1]))
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))

    f_output = open(args.output,'a')
    f_output.write('=======\n')
    f_output.write('{}\n'.format(args.model_name))
    f_output.write('Overall Train score: {}\n'.format(train_score[0]))
    f_output.write('Overall Train accuracy: {}\n'.format(train_score[1]))
    f_output.write('Overall Test score: {}\n'.format(test_score[0]))
    f_output.write('Overall Test accuracy: {}\n'.format(test_score[1]))
    f_output.write("Duration : {}".format(timedelta(seconds=end_time - start_time)))
    f_output.write('=======\n')
    f_output.close()

if __name__ == "__main__":
    main()
