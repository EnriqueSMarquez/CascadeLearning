# CascadeLearning
This repository contains the scripts used to generate the results on the paper Deep Cascade Learning submitted to Neural Computation Journal. Most results and scripts are made for cifar10 and cifar100.

theAllCNN_Cascade.py, cascades The All CNN network and also trains the end-end version, saves the results in the folder specified on the script (default TheAllCNN_WithDenseBlock)

VGG_Cascade.py, cascades a VGG style network and also trains the end-end version, saves the results in the folder specified on the script (default VGG_Results)

getWeightsImage.py, get the image of the first layer weights given the model

mnistMLPCascade.py, cascade a small backprop problem using MNIST dataset and a three MLP network

timeComplexityTest.py, cascades a VGG network multiple times, with different initial epochs

usefulMethods.py, methods that are used across the scripts in this repository