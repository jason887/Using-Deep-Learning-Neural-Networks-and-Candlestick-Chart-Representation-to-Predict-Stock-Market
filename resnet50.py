import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


# dimensions of our images.
img_width, img_height = 200, 200
channel = 3
epochs = 60
batch_size = 64
SHAPE = (img_width, img_height ,channel)
bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

data_directory = "data/dataset_white/"

print ("loading dataset")

X, y, tags = dataset.dataset(data_directory, img_width)
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

def build_model(seed=None):
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
    #print(x)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model

if __name__ == "__main__":

    model = build_model()
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs)

    # Save Model or creates a HDF5 file
    model.save('60_resnet50_efflux_model.h5', overwrite=True)
    #del model  # deletes the existing model



    # predict
    pred_y = model.predict(X_test)

    print(pred_y);

    score = model.evaluate(X_test, Y_test, verbose=1)
    print('\n\nOverall Test score:', score[0])
    print('Overall Test accuracy:', score[1])
