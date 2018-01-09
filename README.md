# CascadeLearning
This repository contains the scripts used to generate the results on the paper Deep Cascade Learning submitted to Neural Computation Journal. Most results and scripts are made for cifar10 and cifar100.

theAllCNN_Cascade_pretraining.py, cascades The All CNN network and also trains the end-end version, saves the results in the folder specified on the script. If specified it can also fine-tune the model after cascading the network.

VGG_Cascade_pretraining.py, cascades a VGG style network and also trains the end-end version, saves the results in the folder specified on the script. If specified it can also fine-tune the model after cascading the network.

VGG_Cascade_with_loop.py, similar to VGG_Cascade_pretraining.py. It has a main loop to tune or test using multiple parameters (epochs, number of units, etc).

VGG_cascade_gradients.py, similar to VGG_Cascade_pretraining.py. It also computes the gradients of the cascade learning and the end-end.

train_cascaded_model.py, trains cascaded model already saved on specified path.

getWeightsImage.py, get the image of the first layer weights given the model

mnistMLPCascade.py, cascade a small backprop problem using MNIST dataset and a three MLP network

timeComplexityTest.py, cascades a VGG network multiple times, with different initial epochs

utils.py, methods that are used across the scripts in this repository

