import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# maximun alloc gpu50% of MEM
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# allocate dynamically
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.layers import Input

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model_mobilenet.h5'
train_data_dir = 'dataset/5/training'
validation_data_dir = 'dataset/5/testing'
nb_train_samples = 4144
nb_validation_samples = 224
epochs = 100
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.MobileNet(include_top=False, weights='imagenet',input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train_mobilenet', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation_mobilenet',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train_mobilenet.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('bottleneck_features_validation_mobilenet.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    train_score = model.evaluate(train_data, train_labels, verbose=1)
    # print('Overall Train score: {}'.format(train_score[0]))
    # print('Overall Train accuracy: {}'.format(train_score[1]))

    test_score = model.evaluate(validation_data, validation_labels, verbose=1)
    # print('Overall Test score: {}'.format(test_score[0]))
    # print('Overall Test accuracy: {}'.format(test_score[1]))

    f_output = open("{}_5.txt".format(top_model_weights_path),'a')
    f_output.write('=======\n')
    f_output.write('Overall Train score: {}\n'.format(train_score[0]))
    f_output.write('Overall Train accuracy: {}\n'.format(train_score[1]))
    f_output.write('Overall Test score: {}\n'.format(test_score[0]))
    f_output.write('Overall Test accuracy: {}\n'.format(test_score[1]))
    f_output.write('=======\n')
    f_output.close()


save_bottlebeck_features()
train_top_model()
