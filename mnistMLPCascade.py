"""
    SMALL BACKPROP PROBLEM USING THE MNIST DATASET
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import cPickle
import numpy as np
from keras.models import model_from_json
from keras.utils.visualize_util import plot
from keras.models import Model
from keras.datasets import mnist
from usefulMethods import LearningRateC

weightDecay = 10e-4
lr = 0.01
sgd = SGD(lr=lr, momentum=0.9)
stringOfHistory = './MNIST_MLP/mlp2layers'

def trainModel(modelToTrain,data,currentEpochs,threshold):
    learningCall = LearningRateC(data[2],data[3],threshold)
    modelToTrain.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, verbose=2,callbacks=[learningCall])
    hist = dict()
    hist['accuracyTraining'] = trainingResults.history['acc']
    hist['lossTraining'] = trainingResults.history['loss']
    hist.update(learningCall.history)
    return hist
def getNewInputs(model,data):
    return (model.predict(data[0]),data[1],model.predict(data[2]),data[3])

def connectOutputBlock(modelToConnectOut):
    modelToConnectOut.add(Dense(256,init='he_normal',W_regularizer=l2(weightDecay)))
    modelToConnectOut.add(Activation('relu'))
    modelToConnectOut.add(Dropout(0.5))
    modelToConnectOut.add(Dense(10,init='he_normal',W_regularizer=l2(weightDecay)))
    modelToConnectOut.add(Activation('softmax'))
    return modelToConnectOut

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = np.expand_dims(X_train,1)
X_test = np.expand_dims(X_test,1)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)

datagen.fit(X_train)

tmpX = list()
tmpY = list()
progbar = generic_utils.Progbar(len(X_train))
print('LOADING TRAINING DATA')
for X_batch, Y_batch in datagen.flow(X_train, Y_train,batch_size=1):
    tmpX.append(X_batch[0,:])
    tmpY.append(Y_batch[0,:])
    progbar.add(1)  
    if(len(tmpX) >= len(X_train)):
        break

X_train = np.asarray(tmpX)
Y_train = np.asarray(tmpY)

tmpX = list()
tmpY = list()
progbar = generic_utils.Progbar(len(X_test))
print('\nLOADING TESTING DATA')
for X_batch, Y_batch in datagen.flow(X_test, Y_test,batch_size=1):
    tmpX.append(X_batch[0,:])
    tmpY.append(Y_batch[0,:])
    progbar.add(1)  
    if(len(tmpX) >= len(X_test)):
        break

X_test = np.asarray(tmpX)
Y_test = np.asarray(tmpY)
del tmpX,tmpY

#TRAINING FIRST LAYER
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1::]))
model.add(Dense(512,init='he_normal',W_regularizer=l2(weightDecay),name='mainLayer'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model = connectOutputBlock(model)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

currentData = (X_train,Y_train,X_test,Y_test)
history = dict()
history['iter0'] = dict()
history['iter0'].update(trainModel(model,currentData,50,22))

layersFirstIter = model.layers[0:3]
model = Sequential()
for currentLayer in layersFirstIter:
	model.add(currentLayer)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
currentData = getNewInputs(model,currentData)

model = Sequential()
model.add(Dense(512,input_dim=currentData[0].shape[1],init='he_normal',W_regularizer=l2(weightDecay)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model = connectOutputBlock(model)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history['iter1'] = dict()
history['iter1'].update(trainModel(model,currentData,50,22))            

print('TRAINING END TO END')
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1::]))
model.add(Dense(512,init='he_normal',W_regularizer=l2(weightDecay)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512,init='he_normal',W_regularizer=l2(weightDecay)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,init='he_normal',W_regularizer=l2(weightDecay)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history['normalTraining'] = trainModel(model,(X_train,Y_train,X_test,Y_test),300,75)
with open(stringOfHistory + '.txt','w') as fp:
    cPickle.dump(history,fp)