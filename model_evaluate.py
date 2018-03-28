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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import time
from datetime import timedelta

from keras.models import load_model

def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes

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
    X_train, Y_train, nb_classes= build_dataset("{}/training".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes= build_dataset("{}/testing".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))

    # load pre-trained model
    model = load_model(args.model_name)

    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted,axis=1)
    Y_test = np.argmax(Y_test, axis = 1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
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

    f_output = open(args.output,'a')
    f_output.write('=======\n')
    f_output.write('{}\n'.format(args.model_name))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()


if __name__ == "__main__":
    main()
