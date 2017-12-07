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


# dimensions of our images.
img_width, img_height = 48, 48
channel = 3
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/training'
validation_data_dir = 'data/validation'
nb_train_samples = 272
nb_validation_samples = 80
epochs = 100
batch_size = 16
SHAPE = (img_width, img_height ,channel)
bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train_resnet50',bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation_resnet50',bottleneck_features_validation)

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
    #x = AveragePooling2D((7, 7), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(2, activation='softmax', name='fc10')(x)

    model = Model(input_layer, x)

    return model

if __name__ == "__main__":
    save_bottlebeck_features()
    train_data = np.load('bottleneck_features_train_resnet50.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('bottleneck_features_validation_resnet50.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=64, nb_epoch=60)
    model.save('resnet50_final.h5')
    score = model.evaluate(validation_data=val_batches, verbose=1)
    print('\n\nOverall Test score:', score[0])
    print('Overall Test accuracy:', score[1])
