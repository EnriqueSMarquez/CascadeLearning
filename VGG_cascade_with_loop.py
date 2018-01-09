"""
  CASCADES VGG STYLE NET AND TRAINS THE END-END MODEL
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.merge import Concatenate

from keras.regularizers import l2
import cPickle
import numpy as np
import time
import shutil
# from keras.utils.visualize_util import plot
from keras.models import Model,model_from_json
from keras import backend as K
import os
from utils import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadeTraining, save_full_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# np.random.seed(7)
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,device_count={'CPU': 1})
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# sess.as_default()
# K.set_session(sess)

def make_parallel(model, devices):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in devices:
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on `this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':len(devices)})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))
            
        return Model(inputs=model.inputs, outputs=merged)

#GET THE VGG MODEL USED IN THIS TEST
def get_vgg_style_net(input_shape=(32,32,3),blocks=[3,3],outNeurons=512,init_filters=128,dropout=False,nb_classes=10):
  inputs = Input(shape=input_shape)
  x = Conv2D(init_filters,3,kernel_regularizer=l2(weightDecay),padding='same',activation='relu')(inputs)
  blocks[0] -= 1
  for current_block in blocks:
    for current_layer_in_block in range(current_block):
      x = Conv2D(init_filters,3,kernel_regularizer=l2(weightDecay),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    init_filters *= 2
    if dropout:
      x = Dropout(0.5)(x)
  x = Flatten()(x)
  if not dropout:
    x = Dropout(0.5)(x)
  x = Dense(outNeurons,kernel_regularizer=l2(weightDecay),activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(outNeurons/2,kernel_regularizer=l2(weightDecay),activation='relu')(x)
  out = Dense(nb_classes,kernel_regularizer=l2(weightDecay),activation='softmax')(x)

  return Model(inputs=inputs,outputs=out)

def plot_history(history,savingFolder,neurons_str='512'):
  results = dict()
  if not os.path.isdir(savingFolder+'plots'):
    os.mkdir(savingFolder+'plots')
  results['accuracyTraining'] = list()
  results['lossTraining'] = list()
  results['accuracyTest'] = list()
  results['lossTest'] = list()
  time_cascade = list()
  for i in range(len(history.keys())-2):
    results['accuracyTraining'].extend(history['iter' + str(i)]['accuracyTraining'])
    results['lossTraining'].extend(history['iter' + str(i)]['lossTraining'])
    results['accuracyTest'].extend(history['iter' + str(i)]['accuracyTest'])
    results['lossTest'].extend(history['iter' + str(i)]['accuracyTest'])
    time_cascade.extend(history['iter' + str(i)]['time'])
  time_cascade = np.hstack(time_cascade)

  for i in range(len(time_cascade)-1):
    time_cascade[i+1] += time_cascade[i]
  for i in range(len(history['normalTraining']['time'])-1):
    history['normalTraining']['time'][i+1] += history['normalTraining']['time'][i]

  #VS TIME
  for current_data_string in results.keys():
    plt.figure()
    plt.plot(time_cascade,results[current_data_string],lw=2,alpha=3,label='Cascade Training')
    # plt.hold(True)
    plt.plot(history['normalTraining']['time'],history['normalTraining'][current_data_string],lw=2,alpha=3,label='End-to-End')
    plt.legend(loc=0,prop={'size' : 12})
    plt.xlabel('TIME (min)')
    if current_data_string[0].lower() == 'a':
        plt.ylabel('ACCURACY')
    else:
        plt.ylabel('LOSS')
    plt.savefig(savingFolder+ 'plots/' + current_data_string+ '_vs_time_'+neurons_str+'.jpg',bbox_inches='tight',format='jpg',dmi=1000)

    plt.figure()
    plt.plot(results[current_data_string],lw=2,alpha=3,label='Cascade Training')
    # plt.hold(True)
    plt.plot(history['normalTraining'][current_data_string],lw=2,alpha=3,label='End-to-End')
    plt.legend(loc=0,prop={'size' : 12})
    plt.xlabel('EPOCHS')
    if current_data_string[0].lower() == 'a':
        plt.ylabel('ACCURACY')
    else:
        plt.ylabel('LOSS')
    plt.savefig(savingFolder + 'plots/'+current_data_string+ '_vs_epochs_'+neurons_str+'.jpg',bbox_inches='tight',format='jpg',dmi=1000)

batch_size = 128 #BATCH SIZE OF TRAINING/TESTING
nb_classes = 10 #NUMBER OF CLASSES (DEPENDING ON THE DATASET 10 OR 100)
nb_epoch = 10 #NUMBER OF INITIAL EPOCHS FOR THE CASCADE LEARNING
lr = 0.01 #INITIAL LEARNING RATE
weightDecay = 10.e-4 #WEIGHT DECAY OF THE MODEL
doCascade = [True]
# index = int(sys.argv[-1])
saveResults = True #SAVE THE RESULTS IN FOLDER
depth = 256
outNeurons = [256]

# stringOfHistory = '/data/esm1g14/SecondHalf/CascadeLearning/VGG_Results/Run1_gradients/'
# stringOfHistory = './VGG_Results/Run1_out256/'
stringOfHistory = './VGG_Results/Run1_depth_9/'

if not os.path.isdir(stringOfHistory):
    os.mkdir(stringOfHistory)
if not os.path.isfile(stringOfHistory+'run_file.py'):
  shutil.copyfile('./VGG_cascade_with_loop.py',stringOfHistory+'run_file.py')

print(stringOfHistory)
(X_train, y_train), (X_test, y_test) = cifar10.load_data() #GET DATA

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#PRECISION AND PIXEL VALUES IN THE RANGE FROM 0 T#O 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# # LESS DATA TO TEST SCRIPT FASTER
# X_train = X_train[0:100]
# Y_train = Y_train[0:100]
# X_test = X_test[0:100]
# Y_test = Y_test[0:100]

#GET VALITDATION SET FROM TRAINING SET
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]
X_val = X_train[0:int(len(X_train)/10)]
Y_val = Y_train[0:int(len(Y_train)/10)]
X_train = X_train[int(len(X_train)/10)::]
Y_train = Y_train[int(len(Y_train)/10)::]

#DATA GENERATOR USED IN CASE OF DATA AUGMENTATION. ALSO USED TO ENHANCE THE CASCADE LEARNING PROCEDURE
datagen = ImageDataGeneratorForCascading(featurewise_center=True,  #MEAN 0
                                         featurewise_std_normalization=True,  #STD 1
                                         zca_whitening=True,  #WHITENING
                                         rotation_range=False,  # randomly rotate images in the range (degrees, 0 to 180)
                                         width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                                         height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                                         horizontal_flip=False,  # randomly flip images
                                         vertical_flip=False,
                                         modelToPredict=None)  #MODEL USED TO MAKE PREDICTIONS USING ALREADY TRAINED LAYERS

datagen.fit(X_train) #CALCULATE NORMALIZATION AND WHITENING PARAMETERS

# for i,current_image in enumerate(X_train):
#   X_train[i,:,:,:] = datagen.standardize(current_image)

# for i,current_image in enumerate(X_test):
#   X_test[i,:,:,:] = datagen.standardize(current_image)

# for i,current_image in enumerate(X_val):
#   X_val[i,:,:,:] = datagen.standardize(current_image)
# with tf.device('/gpu:0'):
#   X_train = tf.constant(X_train)
#   X_test = tf.constant(X_test)
history = dict()
# epochs_array = [10,30,50,70,100]
# for current_epochs in epochs_array:
# current_epochs = epochs_array[index]
print(('CASCADING WITH %d DEPTH')%(depth))
history[str(depth)] = dict()
if not os.path.isfile(stringOfHistory+'history'+str(depth)+'.txt'):
  sgd = SGD(lr=lr, momentum=0.9) #OPTIMIZER TO USE (STOCHASTIC GRADIENT DESCENT)
  model = get_vgg_style_net(dropout=False,outNeurons=512,nb_classes=nb_classes,blocks=[3,3,3]) #GET THE MODEL

  if saveResults:
    save_full_model(model,history=None,path=stringOfHistory,name='init_model_'+str(depth)+'.h5')

  # print(('CASCADING MODEL WITH INIT EPOCHS : %d ')%(current_epochs))
  model.summary()
  #CASCADE THE MODE9
  cascadedModel, tmp_history = CascadeTraining(model,X_train,Y_train,
                                                dataAugmentation=datagen,
                                                X_val=X_val,Y_val=Y_val,
                                                X_test=X_test,Y_test=Y_test,
                                                stringOfHistory=None,
                                                epochs=10,
                                                loss='categorical_crossentropy',
                                                optimizer=sgd,initialLr=lr,weightDecay=weightDecay,
                                                patience=10,windowSize=5,batch_size=batch_size,
                                                outNeurons=512,
                                                nb_classes=nb_classes,
                                                index=depth,gradient=False,fast=True)
  history[str(depth)].update(tmp_history)
  if saveResults:
    save_full_model(cascadedModel,None,stringOfHistory,name='cascaded_model'+str(depth)+'.h5')
    save_full_model(None,history,stringOfHistory,name=str(depth))
else:
  history = cPickle.load(open(stringOfHistory+'history'+str(depth)+'.txt','r'))
# np.random.seed(7)
history = cPickle.load(open(stringOfHistory + 'history_' + str(depth) + '_.txt','r'))
del history['normalTraining']
model = get_vgg_style_net(dropout=True,outNeurons=256,nb_classes=nb_classes,blocks=[3,3,3]) #GET THE MODEL
# model = make_parallel(model, [1,2])
sgd = SGD(lr=0.01, momentum=0.9) #OPTIMIZER TO USE (STOCHASTIC GRADIENT DESCENT)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen.modelToPredict = None #TRAINING END-END (NO MODEL TO PREDICT IS REQUIRED)
#CALLBACK TO REDUCE THE LEARNING RATE ON PLATEOUS AND SAVE THE VALIDATION/TESTING RESULTS
learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience=75,windowSize=20,gradient=False)
#TRAIN THE END-END MODEL
print('TRAINING END-END MODEL WITH DEPTH ' + str(depth))
tmpHistory = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch=np.ceil(1.*len(X_train)/batch_size).astype(int),epochs=250,verbose=1,callbacks=[learningCall])
# tmpHistory = model.fit(X_train, Y_train, batch_size=batch_size,epochs=250,verbose=1,callbacks=[learningCall])

#OBTAIN FINAL RESULTS
history['normalTraining'] = dict()
history['normalTraining']['lossTraining'] = tmpHistory.history['loss']
history['normalTraining']['accuracyTraining'] = tmpHistory.history['acc']
history['normalTraining'].update(learningCall.history)
history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

  SAVE RESULTS AND MODEL IF NECESSARY
if saveResults:
  save_full_model(model,None,stringOfHistory,'end_end_'+str(depth)+'.h5')
  save_full_model(None,history,stringOfHistory,name=str(depth)+'all')
plot_history(history,stringOfHistory,str(current_epochs))