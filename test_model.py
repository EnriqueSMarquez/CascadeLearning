from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.merge import Concatenate
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import Callback
from keras.regularizers import l2
import cPickle
import numpy as np
import time
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as im
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

load_path = './Run4_Augmentation_No_Flip/'
model = load_model(load_path + 'model.h5')

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)
datagen.fit(X_train)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
tmp = model.evaluate_generator(datagen(X_test,Y_test,batch_size=32),4*len(X_test))
print('RESNET')
print(('TEST LOSS : %.2f')%(tmp[0]))
print(('TEST ACC : %.2f')%(tmp[1]))