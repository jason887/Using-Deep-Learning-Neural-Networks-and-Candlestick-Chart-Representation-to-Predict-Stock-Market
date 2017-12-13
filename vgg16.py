import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train_vgg16',bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation_vgg16',bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train_vgg16.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('bottleneck_features_validation_vgg16.npy')
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

    score = model.evaluate(validation_data, validation_labels, verbose=1)
    print('\n\nOverall Test score:', score[0])
    print('Overall Test accuracy:', score[1])
    model.save_weights(top_model_weights_path)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=200)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs',type=int,  default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-w', '--weight_path',
                        help='top model wights path')
    parser.add_argument('-o', '--optimizer',
                        help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension

    top_model_weights_path = 'bottleneck_fc_model_vgg16.h5'
    train_data_dir = 'data/training'
    validation_data_dir = 'data/validation'
    nb_train_samples = 272
    nb_validation_samples = 80
    epochs = args.epochs
    batch_size = args.batch_size
    save_bottlebeck_features()
    train_top_model()

if __name__ == "__main__":
    main()
