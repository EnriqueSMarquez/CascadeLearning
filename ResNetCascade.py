from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import DirectoryIterator, Iterator, transform_matrix_offset_center, flip_axis, apply_transform
from keras.models import Sequential
from keras.layers import merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import Callback
import cPickle
import numpy as np
import time
# from keras.utils.visualize_util import plot
from keras import backend as K
from scipy import linalg
from keras.models import model_from_json
import os
from keras.utils.visualize_util import plot
from keras.models import Model

batch_size = 128
nb_classes = 10
nb_epoch = 50
lr = 0.01
weightDecay = 10.e-4
initFilters = 64
sgd = SGD(lr=lr, momentum=0.9)
numberOfResBlocks = 6

def resBlock(inputToBlock,filters,stride=(1,1)):
    if stride[0] == 1:
        bn1 = BatchNormalization(axis=1, mode=2)(inputToBlock)
        relu1 = Activation('relu')(bn1)
        conv1 = Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(relu1)
        bn2 = BatchNormalization(axis=1, mode=2)(conv1)
        relu2 = Activation('relu')(bn2)
        conv2 = Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(relu2)
        return merge([inputToBlock,conv2],'sum')
    else:
        bn1 = BatchNormalization(axis=1, mode=2)(inputToBlock)
        relu1 = Activation('relu')(bn1)
        conv1 = Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal',subsample=(2,2))(relu1)
        bn2 = BatchNormalization(axis=1, mode=2)(conv1)
        relu2 = Activation('relu')(bn2)
        conv2 = Convolution2D(2*filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(relu2)

        bypass = Convolution2D(2*filters, 1, 1,W_regularizer=l2(weightDecay),border_mode='valid',init='he_normal',subsample=(2,2))(inputToBlock)
        return merge([bypass,conv2],'sum')

def resBlockWithoutBypass(inputToBlock,filters,stride=(1,1)):
    bn1 = BatchNormalization(axis=1, mode=2)(inputToBlock)
    relu1 = Activation('relu')(bn1)
    conv1 = Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal',subsample=stride)(relu1)
    bn2 = BatchNormalization(axis=1, mode=2)(conv1)
    relu2 = Activation('relu')(bn2)
    if stride[0] == 2:
        filters *= 2
    return Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(relu2)

def convLayer(inputBlock,filters,stride=(1,1)):
    conv1 = Convolution2D(filters, 3, 3,W_regularizer=l2(weightDecay),border_mode='same',init='he_normal',subsample=stride)(inputToBlock)
    bn2 = BatchNormalization(axis=1, mode=2)(conv1)
    return Activation('relu')(bn2)

def connectOutputBlock1(lastLayer):
    flatten = Flatten()(lastLayer)
    dense1 = Dense(256,W_regularizer=l2(weightDecay),init='he_normal')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    act1 = Activation('relu')(dropout1)
    output = Dense(10,W_regularizer=l2(weightDecay),init='he_normal')(act1)
    return Activation('softmax')(output)

def connectOutputBlock2(lastLayer):
    avg = AveragePooling2D(pool_size=(lastLayer._keras_shape[2],lastLayer._keras_shape[3]),strides=(1, 1))(lastLayer)
    flatten = Flatten()(avg)
    return Dense(output_dim=10, init='he_normal', activation='softmax',W_regularizer=l2(weightDecay))(flatten)

def trainModel(modelToTrain,data,currentEpochs,threshold,generator=None):
    learningCall = LearningRateC(data[2],data[3],threshold,generator)
    modelToTrain.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    if generator == None:
        trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, verbose=2,callbacks=[learningCall])
    else:
        trainingResults = modelToTrain.fit_generator(generator.flow(data[0],data[1], batch_size=batch_size),samples_per_epoch=len(data[0]), nb_epoch=currentEpochs, verbose=2,callbacks=[learningCall])
    hist = dict()
    hist['accuracyTraining'] = trainingResults.history['acc']
    hist['lossTraining'] = trainingResults.history['loss']
    hist.update(learningCall.history)
    return hist
def getNewInputs(model,data):
    return (model.predict(data[0]),data[1],model.predict(data[2]),data[3])

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

stringOfHistory = './ResNetResults/ResNet9Blocks_avg'

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,
    modelToPredict=None)  # randomly flip images

datagen.fit(X_train)

# tmpX = list()
# tmpY = list()
# progbar = generic_utils.Progbar(len(X_train))
# print('LOADING TRAINING DATA')
# for X_batch, Y_batch in datagen.flow(X_train, Y_train,batch_size=1):
#     tmpX.append(X_batch[0,:])
#     tmpY.append(Y_batch[0,:])
#     progbar.add(1)  
#     if(len(tmpX) >= len(X_train)):
#         break

# X_train = np.asarray(tmpX)
# Y_train = np.asarray(tmpY)

# tmpX = list()
# tmpY = list()
# progbar = generic_utils.Progbar(len(X_test))
# print('\nLOADING TESTING DATA')
# for X_batch, Y_batch in datagen.flow(X_test, Y_test,batch_size=1):
#     tmpX.append(X_batch[0,:])
#     tmpY.append(Y_batch[0,:])
#     progbar.add(1)  
#     if(len(tmpX) >= len(X_test)):
#         break

# X_test = np.asarray(tmpX)
# Y_test = np.asarray(tmpY)
# del tmpX,tmpY

print('TRAINING END TO END')
history = dict()

inputs = Input(shape=(3,32,32))
inputConv = Convolution2D(initFilters, 3, 3,subsample=(2,2),W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(inputs)
inputBN = BatchNormalization(axis=1, mode=2)(inputConv)
inputAct = Activation('relu')(inputBN)
# inputBlockOut = MaxPooling2D(pool_size=(2,2), strides=(2, 2), border_mode='same')(inputAct)

resBlock1 = resBlock(inputAct,initFilters)
resBlock2 = resBlock(resBlock1,initFilters)
resBlock3 = resBlock(resBlock2,initFilters,(2,2))
resBlock4 = resBlock(resBlock3,2*initFilters)
resBlock5 = resBlock(resBlock4,2*initFilters)
resBlock6 = resBlock(resBlock5,2*initFilters,(2,2))
resBlock7 = resBlock(resBlock6,4*initFilters)
resBlock8 = resBlock(resBlock7,4*initFilters)
resBlock9 = resBlock(resBlock8,4*initFilters,(2,2))

out = connectOutputBlock2(resBlock9)

model = Model(input=inputs, output=out)
plot(model, to_file=stringOfHistory+'.png',show_shapes=True)

history['normalTraining'] = trainModel(model,(X_train,Y_train,X_test,Y_test),nb_epoch*6,75,datagen)
# history['normalTraining']['lossTraining'] = tmpHistory.history['loss']
# history['normalTraining']['accuracyTraining'] = tmpHistory.history['acc']
# history['normalTraining']['lossTest'] = learningCall.history['lossTest']
# history['normalTraining']['accuracyTest'] = learningCall.history['accuracyTest']
# history['normalTraining']['time'] = learningCall.history['time']
# history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

open(stringOfHistory + '.json', 'w').write(model.to_json())
model.save_weights(stringOfHistory + '.h5',overwrite=True)
#CASCADE THE RESNET
#NEED DADAGEN TO BE USED ONLY AT THE BEGGINING. USE FIT ALL THE WAY.
#FIRST LAYER
print('TRAINING FIRST LAYER')
inputs = Input(shape=(3,32,32))
inputConv = Convolution2D(initFilters, 7, 7,subsample=(2,2),W_regularizer=l2(weightDecay),border_mode='same',init='he_normal')(inputs)
inputBN = BatchNormalization(axis=1, mode=2)(inputConv)
inputAct = Activation('relu')(inputBN)
inputMP = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(inputAct)

outputFirstIter = connectOutputBlock1(inputMP)

firstIterModel = Model(input=inputs, output=outputFirstIter)
plot(firstIterModel, to_file='ResNetResults/firstIter.png',show_shapes=True)
currentData = (X_train,Y_train,X_test,Y_test)
history['iter0'] = dict()
# history['iter0'].update(trainModel(firstIterModel,currentData,nb_epoch/5,22))
firstIterModel = Model(input=inputs,output=inputMP)
currentData = getNewInputs(firstIterModel,currentData)

del firstIterModel

filtersToCascade = [initFilters,initFilters,initFilters,2*initFilters,2*initFilters,2*initFilters]
strides = [1,1,2,1,1,2]
epochs = np.linspace(0.4,1,numberOfResBlocks)*nb_epoch
for currentBlock in range(numberOfResBlocks):
    print('TRAINING ' + str(int(currentBlock+1)) + ' LAYER')
    inputs = Input(shape=currentData[0].shape[1::])
    currentResBlock = convLayer(inputs,filtersToCascade[currentBlock],(strides[currentBlock],strides[currentBlock]))
    out = connectOutputBlock1(currentResBlock)
    model = Model(input=inputs, output=out)
    plot(model, to_file='ResNetResults/iter' + str(int(currentBlock+1)) + '.png',show_shapes=True)
    history['iter' + str(int(currentBlock+1))] = dict()
    history['iter' + str(int(currentBlock+1))].update(trainModel(model,currentData,int(epochs[currentBlock]),22))
    model = Model(input=inputs,output=currentResBlock)
    currentData = getNewInputs(model,currentData)
del currentData

# tmpHistory = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,callbacks=[learningCall])

with open(stringOfHistory + '.txt','w') as fp:
    cPickle.dump(history,fp)