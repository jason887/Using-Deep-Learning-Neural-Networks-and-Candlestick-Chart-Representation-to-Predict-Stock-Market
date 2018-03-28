import tensorflow as tf # uncomment this for using GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# maximun alloc gpu50% of MEM
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
#allocate dynamically
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


import tensorflow as tf
import math
import json
import sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label


def build_model(SHAPE, nb_classes, bn_axis, seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
          np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')


    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # print(x)
    # x = AveragePooling2D((7, 7), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model

def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    # print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))
    return num_file

def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=48)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=1)
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
    data_directory = args.input

    SHAPE = (img_width, img_height ,channel)
    nb_train_samples = countImage('{}/training'.format(data_directory))
    nb_validation_samples = countImage('{}/testing'.format(data_directory))
    nb_val_pos = countImage('{}/testing/1'.format(data_directory))
    nb_val_neg = countImage('{}/testing/0'.format(data_directory))
    bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    # early_stopper = EarlyStopping(min_delta=0.001, patience=10)

    period_name = data_directory.split('/')
    nb_classes = 2
    print ("loading dataset")
    X_train, Y_train = build_dataset("{}/training".format(data_directory), args.dimension)
    X_test, Y_test = build_dataset("{}/testing".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))
    model = build_model(SHAPE,nb_classes,bn_axis)
    model.compile(optimizer=args.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # # this is the augmentation configuration we will use for training
    # train_datagen = ImageDataGenerator()
    #
    # # this is the augmentation configuration we will use for testing:
    # # only rescaling
    # test_datagen = ImageDataGenerator()
    #
    # # this is a generator that will read pictures found in
    # # subfolers of 'data/train', and indefinitely generate
    # # batches of augmented image data
    # train_generator = train_datagen.flow_from_directory(
    #         '{}/training'.format(data_directory),  # this is the target directory
    #         target_size=(img_width, img_height),  # all images will be resized to 150x150
    #         batch_size=batch_size,
    #         class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    #
    # # this is a similar generator, for validation data
    # validation_generator = test_datagen.flow_from_directory(
    #         '{}/testing'.format(data_directory),
    #         target_size=(img_width, img_height),
    #         batch_size=batch_size,
    #         class_mode='binary')
    #
    # # Fit the model
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=nb_validation_samples // batch_size)


    # Save Model or creates a HDF5 file
    # model.save('{}epochs_{}period_{}dimension_resnet50_model.h5'.format(epochs,period_name[2],period_name[1]), overwrite=True)
    # del model  # deletes the existing model

    model.save_weights('first_tryx.h5', overwrite=True)
    #print(validation_generator)
    #
    # predicted = model.predict_generator(validation_generator)
    predicted = model.predict(X_test)
    print(predicted)
    #print(predicted)
    # test_eval = model.evaluate_generator(validation_generator)
    # print(test_eval)
    # train_eval = model.evaluate_generator(train_generator)
    # print(train_eval)
    y_true = np.array([0] * nb_val_neg + [1] * nb_val_pos)
    print(y_true)
    # below 50% means positive for binary class
    y_pred = predicted > 0.5
    print(y_pred)
    # report = classification_report(y_true, y_pred)
    # print(report)
    cm = confusion_matrix(y_true, y_pred)
    # predicted = np.argmax(predicted, axis = 1)
    # Y_test = np.argmax(Y_test, axis = 1)
    # cm = confusion_matrix(Y_test, predicted)
    # print(cm)
    tn=cm[0][0]
    fn=cm[1][0]
    tp=cm[1][1]
    fp=cm[0][1]
    if tp==0:
        tp=1
    if tn==0:
        tn=1
    if fp==0:
        fp=1
    if fn==0:
        fn=1
    print("TP {}\nTN {}\nFP {}\nFN {}".format(tp,tn,fp,fn))
    TPR=float(tp)/(float(tp)+float(fn))
    FPR=float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn)),3)
    specitivity=round(float(tn)/(float(tn) + float(fp)),3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)),3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
                                                                (float(tp)+float(fp))
                                                                *(float(tp)+float(fn))
                                                                *(float(tn)+float(fp))
                                                                *(float(tn)+float(fn))
                                                                ),3)
    print("TPR {}\nFPR {}\nACC {}\nSPE {}\nSENS {}\nMCC {}".format(TPR,FPR,accuracy,specitivity,sensitivity,mcc))

    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
    main()
