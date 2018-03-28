import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# maximun alloc gpu50% of MEM
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
#allocate dynamically
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D, AveragePooling2D)
from keras import backend as K
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import dot, multiply, concatenate, add
from keras.utils import np_utils
import dataset
import argparse

import time
from datetime import timedelta

def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    print("number class : {}".format(nb_classes))
    print("sample count : {}".format(sample_count))
    # data_size = sample_count #* 4 // 5
    # # print("data size : {}".format(data_size))
    # X_train = X[:data_size]
    # y_train = y[:data_size]
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # X_test  = X[data_size:]
    # y_test  = y[data_size:]
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X,y,sample_count,nb_classes

def identity_block(input_tensor, kernel_size, filters, stage, block,bn_axis=3):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base +
               '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
               '2c', use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),bn_axis=3):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
               '2c', use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def build_model(SHAPE,nb_classes,bn_axis,seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),bn_axis=bn_axis)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',bn_axis=bn_axis)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',bn_axis=bn_axis)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',bn_axis=bn_axis)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',bn_axis=bn_axis)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',bn_axis=bn_axis)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',bn_axis=bn_axis)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',bn_axis=bn_axis)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',bn_axis=bn_axis)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',bn_axis=bn_axis)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',bn_axis=bn_axis)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',bn_axis=bn_axis)
    #print(x)
    #x = AveragePooling2D((7, 7), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model

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
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-op', '--optimizer',
                        help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='output file report for result', default="output.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height ,channel)
    # print("SHAPE : {}".format(SHAPE))
    bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

    data_directory = args.input
    period_name = data_directory.split('/')

    training_dir = "{}/training".format(data_directory)
    test_dir = "{}/testing".format(data_directory)
    print ("loading training dataset")
    Xtrain, ytrain, sample_count, nb_classes = build_dataset(training_dir, args.dimension)
    X_train = Xtrain[:sample_count]
    y_train = ytrain[:sample_count]
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    print ("loading testing dataset")
    Xtest, ytest, sample_count, test_class = build_dataset(test_dir, args.dimension)
    X_test  = Xtest[:sample_count]
    y_test  = ytest[:sample_count]
    Y_test = np_utils.to_categorical(y_test, test_class)

    model = build_model(SHAPE,nb_classes,bn_axis)

    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Save Model or creates a HDF5 file
    model.save('{}epochs_{}period_{}dimension_resnet18_{}.h5'.format(epochs,period_name[1],args.dimension,args.optimizer), overwrite=True)

    train_score = model.evaluate(X_train, Y_train, verbose=1)
    print('Overall Train score: {}'.format(train_score[0]))
    print('Overall Train accuracy: {}'.format(train_score[1]))

    test_score = model.evaluate(X_test, Y_test, verbose=1)
    print('Overall Test score: {}'.format(test_score[0]))
    print('Overall Test accuracy: {}'.format(test_score[1]))

    f_output = open(args.output,'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}period_{}dimension_resnet18_{}.h5\n'.format(epochs,period_name[1],args.dimension,args.optimizer))
    f_output.write('Overall Train score: {}\n'.format(train_score[0]))
    f_output.write('Overall Train accuracy: {}\n'.format(train_score[1]))
    f_output.write('Overall Test score: {}\n'.format(test_score[0]))
    f_output.write('Overall Test accuracy: {}\n'.format(test_score[1]))
    f_output.write('=======\n')
    f_output.close()

    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
    main()
