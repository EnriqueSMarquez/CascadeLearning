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
import time
# from keras.utils.visualize_util import plot
from keras.models import model_from_json
import os
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadeTraining

#GET THE VGG MODEL USED IN THIS TEST
def getModel():
  model = Sequential()
  model.add(ZeroPadding2D((1,1),input_shape=(3,32,32)))
  model.add(Convolution2D(128, 3, 3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128,3,3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.5))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256,3,3,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(512,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(256,W_regularizer=l2(weightDecay)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5)) 
  model.add(Dense(nb_classes,W_regularizer=l2(weightDecay)))
  model.add(Activation('softmax'))
  return model

batch_size = 64 #BATCH SIZE OF TRAINING/TESTING
nb_classes = 10 #NUMBER OF CLASSES (DEPENDING ON THE DATASET 10 OR 100)
nb_epoch = 10 #NUMBER OF INITIAL EPOCHS FOR THE CASCADE LEARNING
lr = 0.01 #INITIAL LEARNING RATE
weightDecay = 10e-4 #WEIGHT DECAY OF THE MODEL

sgd = SGD(lr=lr, momentum=0.9) #OPTIMIZER TO USE (STOCHASTIC GRADIENT DESCENT)
saveResults = False #SAVE THE RESULTS IN FOLDER
stringOfHistory = './VGG_Results/VGGCascade_Opt_Test' #FOLDER TO SAVE RESULTS IF TRUE

(X_train, y_train), (X_test, y_test) = cifar10.load_data() #GET DATA

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#PRECISION AND PIXEL VALUES IN THE RANGE FROM 0 TO 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#LESS DATA TO TEST SCRIPT FASTER
# X_train = X_train[0:100]
# Y_train = Y_train[0:100]
# X_test = X_test[0:100]
# Y_test = Y_test[0:100]

#GET VALITDATION SET FROM TRAINING SET
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
                                         modelToPredict=None)  #MODEL USED TO MAKE PREDICTIONS USING ALREADY TRAINED LAYERS

datagen.fit(X_train) #CALCULATE NORMALIZATION AND WHITENING PARAMETERS

model = getModel() #GET THE MODEL

#CASCADE THE MODEL
cascadedModel, history = CascadeTraining(model,X_train,Y_train,
                                              dataAugmentation=datagen,
                                              X_val=X_val,Y_val=Y_val,
                                              X_test=X_test,Y_test=Y_test,
                                              stringOfHistory=stringOfHistory,
                                              epochs=nb_epoch,
                                              loss='categorical_crossentropy',
                                              optimizer=sgd,initialLr=lr,weightDecay=weightDecay,
                                              patience=10,windowSize=5,batch_size=batch_size)

model = getModel() #GET MODEL
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen.modelToPredict = None #TRAINING END-END (NO MODEL TO PREDICT IS REQUIRED)

#GET THE NORMALIZED TRAINING SET
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

#CALLBACK TO REDUCE THE LEARNING RATE ON PLATEOUS AND SAVE THE VALIDATION/TESTING RESULTS
learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience=75,windowSize=10)
#TRAIN THE END-END MODEL
tmpHistory = model.fit(tmpX, tmpY, batch_size=batch_size, nb_epoch=300, verbose=2,callbacks=[learningCall])

#OBTAIN FINAL RESULTS
history['normalTraining'] = dict()
history['normalTraining']['lossTraining'] = tmpHistory.history['loss']
history['normalTraining']['accuracyTraining'] = tmpHistory.history['acc']
history['normalTraining'].update(learningCall.history)
history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

#SAVE RESULTS AND MODEL IF NECESSARY
if saveResults:
  open(stringOfHistory + '.json', 'w').write(model.to_json())
  model.save_weights(stringOfHistory + '.h5',overwrite=True)

  with open(stringOfHistory + '.txt','w') as fp:
      cPickle.dump(history,fp)