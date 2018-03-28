'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

# dimensions of our images.
img_width, img_height = 48, 48

train_data_dir = 'cobadata/training'
validation_data_dir = 'cobadata/testing'
nb_train_samples = 1998
nb_validation_samples = 295
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
#print(validation_generator)
#
predicted = model.predict_generator(validation_generator)
#print(predicted)
# test_eval = model.evaluate_generator(validation_generator)
# print(test_eval)
# train_eval = model.evaluate_generator(train_generator)
# print(train_eval)
y_true = np.array([0] * 142 + [1] * 153)
y_pred = predicted > 0.5
report = classification_report(y_true, y_pred)
print(report)
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
