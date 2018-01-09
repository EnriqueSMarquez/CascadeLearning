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
from keras import initializers
from keras.regularizers import l2
import cPickle
import numpy as np
import time
import shutil
# from keras.utils.visualize_util import plot
from keras.models import Model,model_from_json,load_model
import os
from usefulMethods import ImageDataGeneratorForCascading, GetConfusionMatrix,LearningRateC, CascadeTraining_2_Multi_GPU, save_full_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# np.random.seed(7)
import tensorflow as tf
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,device_count={'CPU': 1})
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
# sess.as_default()

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
# K.set_learning_phase(True)

batch_size = 128 #BATCH SIZE OF TRAINING/TESTING
nb_classes = 10 #NUMBER OF CLASSES (DEPENDING ON THE DATASET 10 OR 100)
nb_epoch = 10 #NUMBER OF INITIAL EPOCHS FOR THE CASCADE LEARNING
lr = 0.01 #INITIAL LEARNING RATE
weightDecay = 10.e-4 #WEIGHT DECAY OF THE MODEL
doCascade = [True]
index = int(sys.argv[-1])
saveResults = True #SAVE THE RESULTS IN FOLDER

outNeurons = [64]

# stringOfHistory = '/data/esm1g14/SecondHalf/CascadeLearning/VGG_Results/Run1_time_complexity/'
# stringOfHistory = './VGG_Results/Run1_time_complexity/'
# stringOfHistory = '/data/esm1g14/SecondHalf/CascadeLearning/VGG_Results/Run2_time_complexity_depth_12/'
# stringOfHistory = 
stringOfHistory = ['./VGG_Results/Run1_depth_9/','./The_All_CNN_Results/Run3_pretraining_100/']

# stringOfHistory3 = './VGG_Results/Run2_time_complexity_depth_9/'

depth = [3,10]
# models_to_train = ['cascaded_model10.h5','cascaded_model30.h5','cascaded_model50.h5','cascaded_model70.h5','cascaded_model100.h5']
# histories = ['history10.txt','history30.txt','history50.txt','history70.txt','history100.txt']
models_to_train = ['cascaded_model256.h5','cascaded_model.h5']
histories = ['history256.txt','historycascaded_.txt']
model_path = stringOfHistory[index] + models_to_train[index]
history_path = stringOfHistory[index] + histories[index]
stringOfHistory = stringOfHistory[index]
print('TRAINING CASCADED MODEL AT : ' + model_path)
print('HISTORY AT : ' + history_path)

(X_train, y_train), (X_test, y_test) = cifar10.load_data() #GET DATA

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#PRECISION AND PIXEL VALUES IN THE RANGE FROM 0 T#O 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# LESS DATA TO TEST SCRIPT FASTER
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
sgd = SGD(lr=0.01, momentum=0.9) #OPTIMIZER TO USE (STOCHASTIC GRADIENT DESCENT)
initializer = initializers.get('glorot_uniform')

model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#CASCADED MODEL ACC
tmp = model.evaluate_generator(datagen.flow(X_test, Y_test, batch_size=batch_size),len(X_test)//batch_size)
print(('CASCADED MODEL LOSS : %.3f ACC : %.3f')%(tmp[0],tmp[1]))

for layer in model.layers:
  if isinstance(layer,Dense):
    new_weights = list()
    for weight in layer.get_weights():
      new_weights += [K.eval(initializer(weight.shape))]
    layer.set_weights(new_weights)

tmp = model.evaluate_generator(datagen.flow(X_test, Y_test, batch_size=batch_size),128)
print(('WITH NEW OUTPUT BLOCK LOSS : %.3f ACC : %.3f')%(tmp[0],tmp[1]))

history = cPickle.load(open(history_path,'r'))
# K.set_learning_phase(1)
learningCall = LearningRateC(X_val,Y_val,X_test,Y_test,datagen,batch_size,patience=50,windowSize=20,gradient=False)
#TRAIN THE END-END MODEL
print('TRAINING CASCADED MODEL FROM PATH :' + model_path)

tmpHistory = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch=np.ceil(1.*len(X_train)/batch_size).astype(int),epochs=250,verbose=1,callbacks=[learningCall])
# tmpHistory = model.fit(X_train, Y_train, batch_size=batch_size,epochs=250,verbose=1,callbacks=[learningCall])

#OBTAIN FINAL RESULTS
history['pretrained'] = dict()
history['pretrained']['lossTraining'] = tmpHistory.history['loss']
history['pretrained']['accuracyTraining'] = tmpHistory.history['acc']
history['pretrained'].update(learningCall.history)
# history['normalTraining']['confusionMatrix'] = GetConfusionMatrix(model,X_test,Y_test,datagen)

  #SAVE RESULTS AND MODEL IF NECESSARY
if saveResults:
  save_full_model(model,None,stringOfHistory,'cascaded_pretrained.h5')
  save_full_model(None,history,stringOfHistory,name='_full')
# plot_history(history,stringOfHistory,str(current_epochs))