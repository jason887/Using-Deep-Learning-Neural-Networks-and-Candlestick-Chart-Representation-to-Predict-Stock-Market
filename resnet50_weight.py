import tensorflow as tf
import os
import sys
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import math
from keras import optimizers


def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))
    return int(num_file)


# dimensions of our images.
img_width, img_height = 200, 200

period = 20
datapath = sys.argv[3]
epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
top_model_weights_path = 'imagenet_resnet50_{}_{}_{}_{}.h5'.format(
    period, epochs, batch_size, datapath)
train_data_dir = '{}/train'.format(datapath)
validation_data_dir = '{}/test'.format(datapath)
nb_train_samples = countImage(train_data_dir)
nb_validation_samples = countImage(validation_data_dir)


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the ResNet50 network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        train_generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train_resnet50_{}_{}_{}_{}'.format(
        period, epochs, batch_size, datapath), bottleneck_features_train)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        validation_generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation_resnet50_{}_{}_{}_{}'.format(period, epochs, batch_size, datapath),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(
        'bottleneck_features_train_resnet50_{}_{}_{}_{}.npy'.format(period, epochs, batch_size, datapath))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(
        'bottleneck_features_validation_resnet50_{}_{}_{}_{}.npy'.format(period, epochs, batch_size, datapath))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save(top_model_weights_path)
    train_score = model.evaluate(train_data, train_labels, verbose=1)
    # print('Overall Train score: {}'.format(train_score[0]))
    # print('Overall Train accuracy: {}'.format(train_score[1]))

    test_score = model.evaluate(validation_data, validation_labels, verbose=1)
    # print('Overall Test score: {}'.format(test_score[0]))
    # print('Overall Test accuracy: {}'.format(test_score[1]))
    f_output = open("outpuresult.txt", 'a')
    f_output.write('=======\n')
    f_output.write('Overall Train score: {}\n'.format(train_score[0]))
    f_output.write('Overall Train accuracy: {}\n'.format(train_score[1]))
    f_output.write('Overall Test score: {}\n'.format(test_score[0]))
    f_output.write('Overall Test accuracy: {}\n'.format(test_score[1]))
    f_output.write('=======\n')
    f_output.close()


save_bottlebeck_features()
train_top_model()
