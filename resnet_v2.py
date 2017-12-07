import pandas as pd
import numpy as np
from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers.core import Flatten, RepeatVector, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import dot, multiply, concatenate, add
from keras.layers.normalization import BatchNormalization


def identity_block(input_tensor, kernel_size, filters, stage, block):
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

    x = ll.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
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

    x = ll.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_block(input_tensor, final_layer_output=220, append='n'):
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1' + append)(input_tensor)
    x = BatchNormalization(name='bn_conv1' + append)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = conv_block(x, 3, [64, 64, 256],
                   stage=2, block='a' + append, strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b' + append)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c' + append)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d' + append)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w' + append)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + append)
    x = AveragePooling1D(final_layer_output, name='avg_pool' + append)(x)
    x = Flatten()(x)
    return x
