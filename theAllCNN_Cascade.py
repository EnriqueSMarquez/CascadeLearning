"""
  CASCADES THE ALL CNN AND TRAINS THE END-END MODEL
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
import cPickle
import numpy as np
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadeTraining

#GET THE ALL CNN MODEL WITH DENSE LAYERS
def getModel1():
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
    model.add(Convolution2D(384,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(384,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(384,3,3,subsample=(2,2),W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(384,3,3,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(outNeurons,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outNeurons/2,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,W_regularizer=l2(weightDecay)))
    model.add(Activation('softmax'))
    return model

#GET THE ALL CNN MODEL WITH DENSE LAYERS
def getModel2():
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
    model.add(Dense(outNeurons,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outNeurons/2,W_regularizer=l2(weightDecay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,W_regularizer=l2(weightDecay)))
    model.add(Activation('softmax'))
    return model

batch_size = 128 #BATCH SIZE OF TRAINING AND TESTING
nb_classes = 10 #NUMBER OF CLASSES
nb_epoch = 50 #NUMBER OF EPOCHS DURING CASCADE TRAINING
lr = 0.01 #INITIAL LEARNING RATE
weightDecay = 10.e-4 #WEIGHT DECAY OF THE TRAINING PROCEDURE
sgd = SGD(lr=lr, momentum=0.9) #OPTIMIZER
saveResults = False

(X_train, y_train), (X_test, y_test) = cifar10.load_data() #LOAD DATA

stringOfHistory = './TheAllCNN_Results/theAllCNN_BiggerNet_NoDropout'
print(stringOfHistory)
Y_train = np_utils.to_categorical(y_train, nb_classes) #CONVERT CLASS VECTORNS INTO AN OUTPUT MATRIX
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
outNeurons = 256

#SMALLER SET FOR DEBUGGING PURPOSES
# X_train = X_train[0:100]
# Y_train = Y_train[0:100]
# X_test = X_test[0:100]
# Y_test = Y_test[0:100]

#GET VALIDATION DATA FROM TRAINING SET
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]
X_val = X_train[0:int(len(X_train)/10)]
Y_val = Y_train[0:int(len(X_train)/10)]
X_train = X_train[int(len(X_train)/10)::]
Y_train = Y_train[int(len(Y_train)/10)::]

#DATA GENERATOR USED IN CASE OF DATA AUGMENTATION. ALSO USED TO ENHANCE THE CASCADE LEARNING PROCEDURE
datagen = ImageDataGeneratorForCascading(featurewise_center=True,  #MEAN 0
                                         featurewise_std_normalization=True,  #STD 1
                                         zca_whitening=True,  #WHITENING
                                         modelToPredict=None) #NECESSARY FOR THE CASCADE LEARNING

datagen.fit(X_train) #CALCULATE NORMALIZATION AND WHITENING PARAMETERS

model = getModel2() #GET MODEL

#CASCADE THE MODEL
modelToTune, history = CascadeTraining(model,X_train,Y_train,
                                          dataAugmentation=datagen,
                                          X_val=X_val,Y_val=Y_val,
                                          X_test=X_test,Y_test=Y_test,
                                          stringOfHistory=stringOfHistory,
                                          epochs=nb_epoch,
                                          loss='categorical_crossentropy',
                                          optimizer=sgd,initialLr=lr,weightDecay=weightDecay,
                                          patience=10,windowSize=5,batch_size=batch_size,
                                          dropout=False)

# history = dict() #INIT DICTIONARY OF RESULTS

sgd = SGD(lr=lr, momentum=0.9) 
model = getModel2() #GET MODEL
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen.modelToPredict = None

#GET NORMALIZED INPUTS
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

#CALLBACK TO REDUCE THE LEARNING RATE ON SCHEDULE AND SAVE THE VALIDATION/TESTING RESULTS
learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience=75,windowSize=10,schedule=[200,250,300])
#TRAIN THE END-END MODEL
tmpHistory = model.fit(tmpX, tmpY, batch_size=batch_size, nb_epoch=350, verbose=2,callbacks=[learningCall])
#GET CONFUSION MATRIX AND OTHER RESULTS OF MODEL THAT HAS BEEN TRAINED
history['normalTraining'] = dict()
history['normalTraining']['lossTraining'] = tmpHistory.history['loss']
history['normalTraining']['accuracyTraining'] = tmpHistory.history['acc']
history['normalTraining'].update(learningCall.history)
history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

#SAVE RESULTS IF NECESSARY
if saveResults:
    open(stringOfHistory + '.json', 'w').write(model.to_json())
    model.save_weights(stringOfHistory + '.h5',overwrite=True)

    with open(stringOfHistory + '.txt','w') as fp:
        cPickle.dump(history,fp)