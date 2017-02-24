"""
    CASCADES THE ALL CNN WITH MULTIPLE TIMES WITH DIFFERENT STARTING EPOCHS
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
import cPickle
import numpy as np
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadeTraining

batch_size = 64 #BATCH SIZE OF TRAINING
nb_classes = 10 #NUMBER OF CLASSES IN DATASET
nb_epoch = 20 #EPOCHS IN TRAINING
lr = 0.01 #INITIAL LEARNING RATE
weightDecay = 10e-4 #WEIGHT DECAY OF THE TRAINING PROCEDURES
sgd = SGD(lr=lr, momentum=0.9) #OPTIMIZER

#THE ALL CNN MODEL
def getModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,32,32),name='z1'))
    model.add(Convolution2D(96,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='z2'))
    model.add(Convolution2D(96,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='z3'))
    model.add(Convolution2D(96,3,3,subsample=(2,2),W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1),name='z4'))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='z5'))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='z6'))
    model.add(Convolution2D(192,3,3,subsample=(2,2),W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1),name='z7'))
    model.add(Convolution2D(192,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Flatten())
    return model

(X_train, y_train), (X_test, y_test) = cifar10.load_data() #GET THE DATA

folderToSaveResults = './TheAllCNN_TimeComplexityTest/'
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#DATA GENERATOR USED IN CASE OF DATA AUGMENTATION. ALSO USED TO ENHANCE THE CASCADE LEARNING PROCEDURE
datagen = ImageDataGeneratorForCascading(featurewise_center=True,  #MEAN 0
                                         featurewise_std_normalization=True,  #STD 1
                                         zca_whitening=True,  #WHITENING
                                         modelToPredict=None) 
datagen.fit(X_train) #GET NORMALIZATION/AUGMENTATION PARAMETERS

epochsInRun = [10,20,30,40,50] #INTIAL EPOCHS ON EVERY RUN OF THE CASCADE LEARNING
for epochs in epochsInRun: #RUN THE CASCADE LEARNING FOR ALL THE GIVEN EPOCHS
    #STATEMENT TO CHECK IF THE CURRENT RUN HAS ALREADY FINISHED
    if not os.path.isdir(folderToSaveResults + str(int(epochs))):
        os.mkdir(folderToSaveResults + str(int(epochs))) #CREATE FOLDER TO STORE RESULTS
        currentFolder = folderToSaveResults + str(int(epochs)) + '/info' #CURRENT RESULTS PATH
        model = getModel() #GET THE ALL CNN MODEL
        #CASCADE THE ALL CNN FOR THE CURRENT RUN
        trainedModel, history = CascadeTraining(model,X_train,Y_train,currentFolder,
                                                      dataAugmentation=datagen,
                                                      X_val=X_test,Y_val=Y_test,
                                                      epochs=epochs,
                                                      loss='categorical_crossentropy',
                                                      optimizer=sgd,initialLr=lr)
        #SAVE RESULTS
        with open(currentFolder + '.txt','w') as fp:
            cPickle.dump(history,fp)
        open(currentFolder + '.json', 'w').write(trainedModel.to_json())
        trainedModel.save_weights(currentFolder + '.h5',overwrite=True)