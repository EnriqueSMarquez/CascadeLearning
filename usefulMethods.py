from keras.preprocessing.image import DirectoryIterator, Iterator, transform_matrix_offset_center, flip_axis, apply_transform
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import os
import threading
import time
from keras import backend as K
from keras.callbacks import Callback
from keras.regularizers import l2
from keras.utils import generic_utils
from keras.models import model_from_json
import cPickle
import time

class LearningRateC(Callback):
    """Callback to reduce the learning rate when validation accuracy plateus,
    save the results on validation and testing sets after every epoch,
    prints the testing results after they are computed, keeps track of
    the time complexity of every epoch, allows to use a schedule if necessary.

        # Arguments
            valX: validation inputs.
            valY: validation targets.
            testX: testing inputs.
            testY: testing targets.
            datagen: data generator (may contain the model to make 
                     predictions on and normalization/augmentation parameters)
            batch_size: integer of images to propagate when testing.
            nb_epoch: integer, the number of epochs to train the model.
            patience: number of epochs to wait until checking whether the validation
                      accuracy has stabilized or not
            windowSize: window size to check plateus
            schedule: schedule to drop learning rate on specific epochs (optional)
    """
    def __init__(self,valX,valY,testX,testY,datagen,batch_size,patience,windowSize,schedule=None):
        self.valX = valX
        self.valY = valY
        self.patience = patience
        self.testX = testX
        self.testY = testY
        self.datagen = datagen
        self.batch_size = batch_size
        self.windowSize = windowSize
        #INIT HISTORY DICTIONARY TO STORE RESULTS
        self.history = dict()
        self.history['accuracyTest'] = list()
        self.history['lossTest'] = list()
        self.history['accuracyValidation'] = list()
        self.history['lossValidation'] = list()
        self.history['time'] = list()
        self.counter = 0 #COUNTER TO KEEP TRACK OF EPOCHS AFTER LEARNING RATE CHANGE   
        self.learningChanges = 0 #NUMBER OF LEARNING RATE CHANGES
        self.schedule = schedule #DROP LEARNING RATE ON SCHEDULE     
    def on_train_begin(self,logs={}):
        pass
    def on_epoch_begin(self,epoch,logs={}):
        self.t0 = time.time() #START TIMER ON EPOCH
    def on_epoch_end(self,epoch,logs={}):
        self.history['time'].append((time.time() - self.t0)/60.) #CHECK TIMER AFTER EPOCH
        tmpAcc = list()
        tmpLoss = list()
        #COMPUTE ACCURACY AND LOSS OF TESTING SET
        for X_batch, Y_batch in self.datagen.flow(self.testX,self.testY,batch_size=self.batch_size):
            tmp = self.model.test_on_batch(X_batch,Y_batch)
            tmpLoss.append(tmp[0])
            tmpAcc.append(tmp[1])
            if len(tmpLoss)*self.batch_size >= len(self.testX):
                break
        self.history['lossTest'].append(np.mean(tmpLoss)) #STORE TESTING LOSS OF CURRENT EPOCH
        self.history['accuracyTest'].append(np.mean(tmpAcc)) #STORE TESTING ACCURACY OF CURRENT EPOCH
        print('TEST LOSS : %.2f TEST ACCURACY : %.2f' % (np.mean(tmpLoss),np.mean(tmpAcc))) #PRINT TESTING RESULTS
        
        tmpAcc = list()
        tmpLoss = list()
        #COMPUTE ACCURACY AND LOSS OF VALIDATION SET
        for X_batch, Y_batch in self.datagen.flow(self.valX,self.valY,batch_size=self.batch_size):
            tmp = self.model.test_on_batch(X_batch,Y_batch)
            tmpLoss.append(tmp[0])
            tmpAcc.append(tmp[1])
            if len(tmpLoss)*self.batch_size >= len(self.testX):
                break

        self.history['lossValidation'].append(np.mean(tmpLoss)) #STORE VALIDATION LOSS
        self.history['accuracyValidation'].append(np.mean(tmpAcc)) #STORE TESTING LOSS
        self.counter += 1
        #IF THERE HAS BEEN ENOUGH EPOCHS TO CHECK IF VALIDATION ACCURACY HAS STABILIZED
        if self.counter > self.patience and self.schedule == None:
            #IF VALIDATION ACCURACY HAS STABILIZED THEN DIVIDE LEARNING RATE BY 10
            if(np.mean(self.history['accuracyValidation'][-self.windowSize::]) <= np.mean(self.history['accuracyValidation'][-2*self.windowSize:-self.windowSize])):
                print('LEARNING RATE IS GOING TO CHANGE')
                K.set_value(self.model.optimizer.lr, self.model.optimizer.lr.get_value()*0.1)
                self.learningChanges += 1
                self.counter = 0
            #IF THERES MORE THAN 3 LEARNING CHANGES, STOP THE TRAINING
            if(self.learningChanges > 3):
                print('BREAK')
                self.model.stop_training = True
        #IF SCHEDULE EXISTS
        elif self.schedule != None:
            #IF CURRENT EPOCH IS IN SCHEDULE DROP LEARNING RATE BY 10
            if (epoch+1) in self.schedule:
                print('LEARNING RATE IS GOING TO CHANGE')
                K.set_value(self.model.optimizer.lr, self.model.optimizer.lr.get_value()*0.1)

class NumpyArrayIterator(Iterator):
    """Modified version of keras NumpyArrayIterator to allow
       a model to predict. Used on the cascade learning to 
       enhance the procedure. 
    """
    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg', modelToPredict=None):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.modelToPredict = modelToPredict
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        #IF MODEL TO PREDICT EXISTS, GET ITS OUTPUT GIVEN THE MINIBATCH
        #YIELD ARTIFICIAL OUTPUTS WITH ITS RESPECTIVE TARGETS
        if self.modelToPredict == None:
            if self.y is None:
                return batch_x
            batch_y = self.y[index_array]
            return batch_x, batch_y
        else:
            if self.y is None:
                return self.modelToPredict.predict(batch_x)
            batch_y = self.y[index_array]
            return self.modelToPredict.predict(batch_x), batch_y

class ImageDataGeneratorForCascading(object):
    """Modified version of keras ImageDataGenerator to allow
       a model to predict. Used on the cascade learning to 
       enhance the procedure.
    """
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering='default',
                 modelToPredict = None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale
        self.modelToPredict = modelToPredict

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, modelToPredict=self.modelToPredict)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        return x

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)

def GetConfusionMatrix(modelToTest,testData,labels,datagen,numberOfClasses=10):
    """Method to obtain confusion matrix on a given data

        # Arguments
            modelToTest: model to compute confusion matrix on.
            testData: inputs to used to calculate confusion matrix.
            labels: labels of testData.
            datagen: data generator (may contain the model to make 
                     predictions on and normalization/augmentation parameters)
            numberOfClasses: number of classes in testData
        #Returns
            Confusion Matrix of the given model on the given data
    """
    confusionMatrix = np.zeros((numberOfClasses,numberOfClasses)) #INITIALIZE CONFUSION MATRIX
    counter = 0 #KEEP TRACK OF TESTING INPUTS
    #GET IMAGES AND LABELS FROM DATAGEN GENERATOR GIVEN testData AND labels
    for currentImage, currentLabel in datagen.flow(testData,labels,batch_size=1):
        currentPrediction = modelToTest.predict(currentImage) #PREDICT ON CURRENT INPUT
        #ADD ONE IN THE CORRESPONDED LOCATION OF THE CONFUSION MATRIX
        confusionMatrix[np.argmax(currentPrediction),np.argmax(currentLabel)] += 1
        counter += 1
        if(counter >= len(testData)): #IF ALL THE INPUTS HAVE BEEN PASSED, BREAK THE LOOP
            break
    #CALCULATE AND PRINT ACCURACY USING THE DIAGONAL OF THE CONFUSION MATRIX
    print(('ACCURACY IN CONFUSION MATRIX : %.2f') % (np.sum(np.diagonal(confusionMatrix))/float(len(testData))))
    return confusionMatrix

def CascadeTraining(model,X_train,Y_train,X_test,Y_test,stringOfHistory=None,dataAugmentation=None,
                        X_val=None,Y_val=None,epochs=20,loss='categorical_crossentropy',
                        optimizer='sgd',initialLr=0.01,weightDecay=10e-4,patience=10,
                        windowSize=5,batch_size=128):
    """Method to cascade a given model

        # Arguments
            model: model to cascade.
            X_train: training inputs.
            Y_train: training targets.
            X_test: test inputs.
            Y_test: test targets.
            stringOfHistory: location to save results
            dataAugmentation: data augmentation generator.
            optimizer: optimizer to ise in every training phase.
            initialLr: initial learning rate.
            weightDecay: weight decay of the training function.
            patience, windowSize: parameters used in the callback.
            batch_size: batch size of training
        #Returns
            Results of training (accuracy, loss), and full model once cascaded
    """
    nextModelToTrain = Sequential() #INIT MODEL TO TRAIN
    saveImportLayersIndexes = list() #INIT VARIABLE TO STORE THE INDEXES OF CORE LAYERS
    # weights = list() #
    history = dict()

    if stringOfHistory != None and os.path.isfile(stringOfHistory + '.txt'): #LOAD HISTORY FILE IF IT EXISTS
        history = cPickle.load(open(stringOfHistory + '.txt','r'))
    else: #OTHERWISE INITIALIZE
        history = dict()
    if stringOfHistory != None and os.path.isfile(stringOfHistory + 'Tmp.json'): #LOAD MODEL TO PREDICT
        nextModelToPredict = model_from_json(open(stringOfHistory +'Tmp.json', 'r').read())
        nextModelToPredict.load_weights(stringOfHistory + 'Tmp.h5')
        nextModelToPredict = nextModelToPredict.layers
    else: #OTHERWISE INITIALIZE
        nextModelToPredict = None
    #SAVE IMPORTANT LAYERS INDEXES
    i = 0
    for currentLayer in model.layers: #GET THE INDEX OF CORE LAYERS IN GIVEN MODEL
        if((currentLayer.get_config()['name'][0] == 'c') or (currentLayer.get_config()['name'][0] == 'f')):
            saveImportLayersIndexes.append(i)
        i += 1
    for i in range(len(saveImportLayersIndexes)-1): #UP TO THE FLATTEN LAYER
        if('iter' + str(i) not in history.keys()): #IF THE LAYER HAS NOT BEEN TRAINED
            history['iter' + str(i)] = dict() #INITIALIZE DICTIONARY TO SAVE RESULTS OF CURRENT RUN
            print('ITERATION %d' % (i))
            if(i == 0): #IF ITS THE FIRST ITERATION
                nextModelToTrain = list()
                for j in model.layers[0:saveImportLayersIndexes[i+1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                if(nextModelToTrain[-1].get_config()['name'][0] == 'z'):
                    del nextModelToTrain[-1]
                tmp = Sequential() #CREATE KERAS MODEL
                for j in nextModelToTrain: #APPEND ALL THE NECESSARY LAYERS TO THE MODEL
                    tmp.add(j)
                nextModelToTrain = tmp
                del tmp

                #APPEND OUTPUT BLOCK TO MODEL
                nextModelToTrain.add(Flatten())
                nextModelToTrain.add(Dense(512,W_regularizer=l2(weightDecay)))
                nextModelToTrain.add(Activation('relu'))
                nextModelToTrain.add(Dropout(0.5))
                nextModelToTrain.add(Dense(256,W_regularizer=l2(weightDecay)))
                nextModelToTrain.add(Activation('relu'))
                nextModelToTrain.add(Dropout(0.5))
                nextModelToTrain.add(Dense(10,W_regularizer=l2(weightDecay)))
                nextModelToTrain.add(Activation('softmax'))

            else: #IF IS NOT THE FIRST ITERATION
                nextModelToTrain = list()
                tmp = Sequential()
                for j in nextModelToPredict: #CREATE KERAS MODEL FOR MODEL TO PREDICT
                    tmp.add(j)
                nextModelToPredict = tmp
                del tmp
                nextModelToPredict.compile(loss=loss,optimizer=optimizer,metrics=['accuracy']) #COMPILE MODEL
                if stringOfHistory != None: #IF SAVING IS REQUIRED (IN CASE THE SCRIPT CRASHES)
                    open(stringOfHistory + 'Tmp.json', 'w').write(nextModelToPredict.to_json()) #SAVE MODEL
                    nextModelToPredict.save_weights(stringOfHistory + 'Tmp.h5',overwrite=True)
                #CHECK OUTPUT OF nextModelToPredict AND SET IT TO THE nextModelToTrain AS INPUT
                if(model.layers[saveImportLayersIndexes[i]-1].get_config()['name'][0] == 'z'):
                    nextModelToTrain.append(model.layers[saveImportLayersIndexes[i]-1])
                for k in model.layers[saveImportLayersIndexes[i]:saveImportLayersIndexes[i+1]]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                    nextModelToTrain.append(k)
                if(nextModelToTrain[-1].get_config()['name'][0] == 'z'): #MAKE SURE MODEL DOES NOT END IN A ZEROPADDING LAYER
                    del nextModelToTrain[-1]
                nextShape = nextModelToPredict.predict(X_train[0:2]).shape[1::] #GET INPUT SHAPE OF THE MODEL TO TRAIN
                #SET THE INPUT SHAPE OF MODEL, PRESERVING PREVIOUS CONFIGURATION.
                #IF IT IS A CONVOLUTIONAL LAYER
                if(nextModelToTrain[0].get_config()['name'][0] == 'c'):
                    k = nextModelToTrain[0].get_config()
                    layerToAppend = Convolution2D(k['nb_filter'],
                                                 k['nb_row'],
                                                 k['nb_col'], 
                                                 W_regularizer=k['W_regularizer'],
                                                 border_mode=k['border_mode'],
                                                 input_shape=(nextShape),
                                                 subsample=k['subsample'],
                                                 weights=nextModelToTrain[0].get_weights())
                #IF IT IS A PADDING LAYER
                elif(nextModelToTrain[0].get_config()['name'][0] == 'z'):
                    layerToAppend = ZeroPadding2D((1,1),input_shape=nextShape,name='zCL'+str(i))
                    nextModelToTrain[0] = layerToAppend
                #IF IT IS A DENSE LAYER
                elif(nextModelToTrain[0].get_config()['name'][0:-2] == 'dense'):
                    k = nextModelToTrain[0].get_config()
                    nextShape = nextModelToPredict.predict(X_train[0:2]).shape[1::][0]
                    layerToAppend = Dense(k['output_dim'],input_dim=nextShape)
                    nextModelToTrain[0] = layerToAppend
            
                #IF THE OUTPUT HAS NOT BEEN FLATTENED 
                if((nextModelToTrain[-1].get_config()['name'][0:-2] != 'flatten') and (nextModelToTrain[0].get_config()['name'][0:-2] != 'dense') and (nextModelToTrain[0].get_config()['name'][0:-3] != 'dense')):
                    nextModelToTrain.append(Flatten())
                #IF OUTPUT BLOCK HAS NOT BEEN CONNECTED
                if((nextModelToTrain[0].get_config()['name'][0:-2] != 'dense') and (nextModelToTrain[0].get_config()['name'][0:-3] != 'dense')):
                    k = nextModelToTrain[0].get_config()
                    nextModelToTrain.append(Dense(512,W_regularizer=l2(weightDecay)))
                    nextModelToTrain.append(Activation('relu'))
                    nextModelToTrain.append(Dropout(0.5))
                    nextModelToTrain.append(Dense(256,W_regularizer=l2(weightDecay)))
                    nextModelToTrain.append(Activation('relu'))
                    nextModelToTrain.append(Dropout(0.5))
                nextModelToTrain.append(Dense(10,W_regularizer=l2(weightDecay)))
                nextModelToTrain.append(Activation('softmax'))

                #INITIALIZE KERAS MODEL USING LAYERS IN nextModelToTrain LIST
                tmp = Sequential()
                for k in nextModelToTrain:
                    tmp.add(k)
                nextModelToTrain = tmp
                del tmp
            K.set_value(optimizer.lr,initialLr) #SET INITIAL LEARNING RATE (IT MIGHT HAVE BEEN CHANGED BY PREVIOUS ITERATIONS)
            nextModelToTrain.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
                
            if nextModelToPredict != None: #IF MODEL TO PREDICT EXISTS
                print('MODEL TO PREDICT LAYERS') #PLOT THE LAYERS OF THE MODEL
                for k in nextModelToPredict.layers:
                    print(k.get_config()['name'])

            print('MODEL TO TRAIN LAYERS') #PLOT LAYERS OF MODEL TO TRAIN
            for k in nextModelToTrain.layers:
                print(k.get_config()['name'])   
            currentEpochs = epochs + (10*i) #SET THE NUMBER OF EPOCHS OF CURRENT RUN
            if currentEpochs > 50: #MAXIMUM NUMBER OF EPOCHS ON CASCADE LEARNING IS 50
                currentEpochs = 50 
            dataAugmentation.modelToPredict = nextModelToPredict #SET MODEL TO GENERATE ARTIFICIAL INPUTS IN GENERATOR
            tmpX = list()
            tmpY = list()
            progbar = generic_utils.Progbar(len(X_train))
            #LOAD ARITIFICAL INPUTS
            print('LOADING TRAINING DATA')
            for X_batch, Y_batch in dataAugmentation.flow(X_train, Y_train,batch_size=1):
                tmpX.append(X_batch[0,:])
                tmpY.append(Y_batch[0,:])
                progbar.add(1)  
                if(len(tmpX) >= len(X_train)):
                    print('\n')
                    break
            tmpX = np.asarray(tmpX)
            tmpY = np.asarray(tmpY)
            #CALLBACK TO REDUCE THE LEARNING RATE AND STORE INFORMATION OF VALIDATION AND TESTING RESULTS DURING TRAINING
            learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,dataAugmentation,batch_size,patience=patience,windowSize=windowSize)
            #TRAIN THE MODEL
            tmpHistory = nextModelToTrain.fit(tmpX, tmpY, batch_size=batch_size, nb_epoch=currentEpochs, verbose=2,callbacks=[learningCall])

            if(nextModelToPredict == None): #IF MODEL TO PREDICT DOES NOT EXIST
                nextModelToPredict = nextModelToTrain.layers[0:-9] #TAKE THE LAYERS OF nextModelToTrain WITHOUT OUTPUT BLOCK
            else: #OTHERWISE APPEND LAYERS (WITHOUT THE OUTPUT BLOCK) OF nextModelToTrain IN nextModelToPredict
                nextModelToPredict = nextModelToPredict.layers
                nextModelToPredict.extend(nextModelToTrain.layers[0:-9])
            #SAVE RESULTS IN SINGLE DICTIONARY, ALSO CALCULATE THE CONFUSION MATRIX OF THE TRAINED MODEL
            history['iter'+str(i)].update(learningCall.history)
            history['iter'+str(i)]['lossTraining'] = tmpHistory.history['loss']
            history['iter'+str(i)]['accuracyTraining'] = tmpHistory.history['acc']
            history['iter'+str(i)]['confusionMatrix'] = GetConfusionMatrix(nextModelToTrain,X_test,Y_test,dataAugmentation)
            if stringOfHistory != None: #SAVE (IF NECESSARY)
                with open(stringOfHistory + '.txt','w') as fp:
                    cPickle.dump(history,fp)
    #GET WHOLE CASCADED MODEL
    modelToReturn = Sequential()
    for i in nextModelToPredict:
        modelToReturn.add(i)
    for i in nextModelToTrain.layers[0:-9]:
        modelToReturn.add(i)

    return modelToReturn, history #RETURN CASCADED MODEL AND RESULTS OF TRAINING
