from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import DirectoryIterator, Iterator, transform_matrix_offset_center, flip_axis, apply_transform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
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
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadePretraining

def getModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,32,32)))
    model.add(Convolution2D(96,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(96,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(96,3,3,subsample=(2,2),W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(192,3,3,subsample=(2,2),W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,W_regularizer=l2(weightDecay)))
    model.add(Activation('softmax'))
    return model

batch_size = 64
nb_classes = 10
nb_epoch = 20
lr = 0.01
weightDecay = 10e-4
sgd = SGD(lr=lr, momentum=0.9)
saveResults = False
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

stringOfHistory = './TheAllCNN_WithDenseBlock/ResultsTest/theAllCNN_NM_Only_test'

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train[0:100]
Y_train = Y_train[0:100]
X_test = X_test[0:100]
Y_test = Y_test[0:100]

permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]
X_val = X_train[0:int(len(X_train)/10)]
Y_val = Y_train[0:int(len(X_train)/10)]
X_train = X_train[int(len(X_train)/10)::]
Y_train = Y_train[int(len(Y_train)/10)::]

datagen = ImageDataGeneratorForCascading(featurewise_center=True,  # set input mean to 0 over the dataset
                                         featurewise_std_normalization=True,  # divide inputs by std of the dataset
                                         zca_whitening=True,  # apply ZCA whitening
                                         modelToPredict=None)  # randomly flip images

datagen.fit(X_train)

model = getModel()

modelToTune, history = CascadePretraining(model,X_train,Y_train,
                                              dataAugmentation=datagen,
                                              X_val=X_val,Y_val=Y_val,
                                              X_test=X_test,Y_test=Y_test,
                                              stringOfHistory=stringOfHistory,
                                              epochs=nb_epoch,
                                              loss='categorical_crossentropy',
                                              optimizer=sgd,initialLr=lr,weightDecay=weightDecay,
                                              patience=10,windowSize=5,batch_size=batch_size)

#TUNE THE ALL CNN
# sgd = SGD(lr=lr, momentum=0.9)
history = dict()
# with open(stringOfHistory + '.txt',"w") as fp:
#     cPickle.dump(history,fp)

sgd = SGD(lr=lr, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen.modelToPredict = None

#gatherInputs
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
tmpX = np.asarray(tmpX)
tmpY = np.asarray(tmpY)

learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience=75,windowSize=10,schedule=[200,250,300])
tmpHistory = model.fit(tmpX, tmpY, batch_size=batch_size, nb_epoch=350, verbose=2,callbacks=[learningCall])


history['normalTraining'] = dict()
history['normalTraining']['lossTraining'] = tmpHistory.history['loss']
history['normalTraining']['accuracyTraining'] = tmpHistory.history['acc']
history['normalTraining'].update(learningCall.history)
history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

if saveResults:
    open(stringOfHistory + '.json', 'w').write(model.to_json())
    model.save_weights(stringOfHistory + '.h5',overwrite=True)

    with open(stringOfHistory + '.txt','w') as fp:
        cPickle.dump(history,fp)