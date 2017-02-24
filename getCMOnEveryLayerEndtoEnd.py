"""
    CALCULATE THE PERFORMANCE AND CONFUSION MATRIX OF EACH LAYER GIVEN AN ALREADY TRAINED MODEL
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np
import cPickle
from keras.regularizers import l2
from keras.models import model_from_json
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix, LearningRateC

from keras import backend as K

(X_train, y_train), (X_test, y_test) = cifar10.load_data() #LOAD DATA
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#GET VALIDATION FROM TRAINING
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
                                         zca_whitening=True, #WHITENING
                                         modelToPredict=None)

datagen.fit(X_train) #CALCULATE NORMALIZATION AND WHITENING PARAMETERS
#LOCATION OF MODEL TO GET PERFORMANCES OF EACH LAYER (TRAINED END-END)
stringOfHistory = './TheAllCNN_WithDenseBlock/Results1/theAllCNN_NM_Only'

#LOAD MODEL
model = model_from_json(open(stringOfHistory+'.json','r').read())
model.load_weights(stringOfHistory+'.h5')
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#ACCURACY OF THE ALREADY END-END TRAINED MODEL
print('FULL MODEL ACCURACY RESULT : ' + str(model.evaluate_generator(datagen.flow(X_test, Y_test,batch_size=64),X_test.shape[0])[1]))
print(stringOfHistory)
saveResults = False

weightDecay = 10e-4 #WEIGHTS DECAY VALUE OF TRAINING PROCEDURE
nb_epoch = 200
saveImportLayersIndexes = list() #VARIABLE TO SAVE WHERE TO SPLIT THE MODEL
i = 0
windowSize = 10 #WINDOW SIZE FOR CALLBACK
history = dict() #INIT RESULTS VARIABLE
patience = 25 #HOLD UNTIL EPOCH (CALLBACK)
lr = 0.01 #LEARNING RATE OF TRAINING
# history = cPickle.load(open(stringOfHistory+'.txt','r'))
for currentLayer in model.layers: #GET THE INDEX OF THE CORE LAYERS IN THE MODEL (CONV AND FLATTEN)
    if((currentLayer.get_config()['name'][0] == 'z') or (currentLayer.get_config()['name'][0] == 'f')): #flatten
        saveImportLayersIndexes.append(i)
    i += 1
# saveImportLayersIndexes.append(23)
for i in range(len(saveImportLayersIndexes)-1): #MINUS THE FLATTEN LAYER
    if ('NM_Layer' + str(i) not in history.keys()): #IF THE LAYER HAS NOT BEEN TRAINED
        history['NM_Layer' + str(i)] = dict() #INIT DICTIONARY TO STORE CURRENT RUN RESULTS
        print(('ITERATION %d') % (i))
        nextModelToPredict = Sequential() #INIT MODEL TO GENERATE FEATURE MAPS
        for j in model.layers[0:saveImportLayersIndexes[i+1]]: #ADD REQUIRED LAYERS
            nextModelToPredict.add(j)
        if nextModelToPredict.layers[-1].get_config()['name'][0] != 'f': #FLATTEN OUTPUT TO CONNECT DENSE LAYERS
            nextModelToPredict.add(Flatten())
        nextModelToPredict.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        nextShape = nextModelToPredict.predict(X_train[0:1]).shape[1::] #GET INPUT SHAPE OF MODEL TO TRAIN

        #MLP TO TRAIN AND COMPUTE ROBUSTNESS OF FEATURE MAPS
        nextModelToTrain = Sequential()
        nextModelToTrain.add(Dense(512,W_regularizer=l2(weightDecay),input_dim=nextShape[0]))
        nextModelToTrain.add(Activation('relu'))
        nextModelToTrain.add(Dropout(0.5))
        nextModelToTrain.add(Dense(256,W_regularizer=l2(weightDecay)))
        nextModelToTrain.add(Activation('relu'))
        nextModelToTrain.add(Dropout(0.5)) 
        nextModelToTrain.add(Dense(10,W_regularizer=l2(weightDecay)))
        nextModelToTrain.add(Activation('softmax'))

        sgd = SGD(lr=lr,momentum=0.9) #OPTIMIZER
        nextModelToTrain.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

        print('MODEL TO PREDICT LAYERS')
        for k in nextModelToPredict.layers:
            print(k.get_config()['name'])
        print('MODEL TO TRAIN LAYERS')
        for k in nextModelToTrain.layers:
            print(k.get_config()['name'])

        tmpX = list()
        tmpY = list()
        batch_size = 64
        datagen.modelToPredict = nextModelToPredict #SET THE MODEL TO MAKE PREDICTIONS ON TO THE GENERATOR

        progbar = generic_utils.Progbar(len(X_train))
        print('LOADING TRAINING DATA') #GET ARTIFICIAL INPUTS FROM GENERATOR
        for X_batch, Y_batch in datagen.flow(X_train, Y_train,batch_size=1):
            tmpX.append(X_batch[0,:])
            tmpY.append(Y_batch[0,:])
            progbar.add(1)  
            if(len(tmpX) >= len(X_train)):
                break

        tmpX = np.asarray(tmpX)
        tmpY = np.asarray(tmpY)
        print('\n')
        #CALLBACK TO REDUCE THE LEARNING RATE AND STORE INFORMATION OF VALIDATION AND TESTING RESULTS DURING TRAINING
        learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience,windowSize)
        #TRAIN THE MODEL
        tmpHistory = nextModelToTrain.fit(tmpX, tmpY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,callbacks=[learningCall])

        #GET CONFUSION MATRIX AND OTHER RESULTS OF MODEL THAT HAS BEEN TRAINED
        history['NM_Layer' + str(i)]['confusionMatrix'] = GetConfusionMatrix(nextModelToTrain,X_test,Y_test,datagen)
        history['NM_Layer' + str(i)]['lossTraining'] = tmpHistory.history['loss']
        history['NM_Layer' + str(i)]['accuracyTraining'] = tmpHistory.history['acc']
        history['NM_Layer' + str(i)].update(learningCall.history)
        #STORE RESULTS IF NECESSARY
        if saveResults:
            with open(stringOfHistory+'.txt','w') as fp:
                cPickle.dump(history,fp)

print('DONE')