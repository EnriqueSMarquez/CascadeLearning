"""
	SAVE THE FILTERS IN THE VGG MODEL AND SAVE THEM AS AN IMAGE
"""

from __future__ import print_function
import numpy as np
from keras.models import model_from_json
from PIL import Image as im

def plot_filters(layer,saving_path='./weights_layer1.jpg',resolution=None,scale=4):
    weights = layer.get_weights()[0]
    weights_image = weights_image.transpose((3,0,1,2))
    if resolution == None:
        resolution = [int(np.sqrt(len(weights))),int(np.sqrt(len(weights)))]
    weights_image = np.zeros((weights.shape[3],resolution[0]*weights.shape[0],resolution[1]*weights.shape[1],weights.shape[2]))
    x = 0
    y = 0
    for i in range(len(weights)):
        weights_image[:,x:x+weights.shape[1],y:y+weights.shape[2]] = weights[i,:,:,:]
        x += weights.shape[1]
        if x == len(weights)*weights.shape[1]:
            x = 0
            y += weights.shape[2]

    weights_image = np.uint8((weights_image/(2*np.max(np.abs(weights_image)))+0.5)*255)

    weights_image = im.fromarray(np.swapaxes(weights_image,0,2))
    weights_image = np.asarray(weights_image.resize((resolution[0]*weights.shape[1]*scale,resolution[1]*weights.shape[2]*scale), im.NEAREST))

    maxIndex = scale*weights.shape[1]+1 #25
    weights_image = np.insert(imageCL,0,0,axis=0)
    nextIndex = maxIndex
    for i in range(resolution[0]-1):
        weights_image = np.insert(weights_image,nextIndex,0,axis=0)
        nextIndex += maxIndex
    weights_image = np.insert(weights_image,weights_image.shape[0],0,axis=0)

    weights_image = np.insert(weights_image,0,0,axis=1)
    nextIndex = maxIndex
    for i in range(resolution[1]-1):
        weights_image = np.insert(weights_image,nextIndex,0,axis=1)
        nextIndex += maxIndex
    weights_image = np.insert(weights_image,weights_image.shape[1],0,axis=1)

    weights_image = im.fromarray(weights_image)
    weights_image.save(saving_path)


def get_weights_image(weights):
	"""Method used to plot the weights of the first layer in a VGG model

        # Arguments
            weights: array with weights
    """
    #VGG NET HAS 128 = 16 * 8 3X3 FILTERS 
	weightsPicture = np.zeros((3,16*3,8*3)) 
	counterX = 0
	counterY = 0
	#ALLOCATE WEIGHTS IN IMAGE
	for i in range(len(weights)):
	    weightsPicture[:,counterX:counterX+3,counterY:counterY+3] = weights[i,:,:,:]
	    counterX += 3
	    if counterX == 16*3:
	        counterX = 0
	        counterY += 3
	#NORMALIZE WEIGHTS
	weightsPicture = np.uint8((weightsPicture/(2*np.max(np.abs(weightsPicture)))+0.5)*255)

	image = im.fromarray(np.swapaxes(weightsPicture,0,2)) #SWAP AXES AND LOAD IMAGE WITH PIL.IMAGE
	image = image.resize((16*3*4,8*3*4), im.NEAREST) #RESIZE FOR BETTER VISUALIZATION AND KEEP RESOLUTION
	image = np.asarray(image) #GET THE IMAGE AS NUMPY ARRAY

	maxIndex = 13 #SIZE OF EACH RESIZED FILTER
	image = np.insert(image,0,0,axis=0) #INSERT FIRST BLACK HORIZONTAL LINE TO SEPARATE FILTERS
	nextIndex = maxIndex
	for i in range(7): #SEVEN MORE LINES TO GO
	    image = np.insert(image,nextIndex,0,axis=0) #INSERT BLACK HORIZONTAL LINE
	    nextIndex += maxIndex #UPDATE
	image = np.insert(image,image.shape[0],0,axis=0) #INSERT LAST BALCK HORIZONTAL LINE

	image = np.insert(image,0,0,axis=1) #INSERT FIRST BLACK VERTICAL LINE TO SEPARATE FILTERS
	nextIndex = maxIndex
	for i in range(15):
	    image = np.insert(image,nextIndex,0,axis=1) #INSERT BLACK VERTICAL LINE
	    nextIndex += maxIndex
	image = np.insert(image,image.shape[1],0,axis=1) #INSERT LAST BLACK VERTICAL LINE

	return im.fromarray(image) #RETURN FILTERS IMAGE AS A PIL.IMAGE OBJECT

modelCL = model_from_json(open('./VGG_CL_Networks/model_CL_Layer0.json','r').read()) #LOAD CASCADED VGG NET
modelCL.load_weights('./VGG_CL_Networks/model_CL_Layer0_weights.h5')

modelEE = model_from_json(open('./VGG_CL_Networks/modelVGG5_End_End.json','r').read()) #LOAD END-END VGG TRAINED NET
modelEE.load_weights('./VGG_CL_Networks/modelVGG5_End_End_weights.h5')

weightsCL = modelCL.layers[1].get_weights()[0] #GET WEIGHTS OF FIRST LAYER
weightsEE = modelEE.layers[1].get_weights()[0]

imageWeightsCL = getWeightsImage(weightsCL) #GET WEIGHTS IMAGES
imageWeightsEE = getWeightsImage(weightsEE)

imageWeightsCL.save('./VGG_CL_Networks/CLWeightsVis.jpg') #SAVE IMAGES
imageWeightsEE.save('./VGG_CL_Networks/EEWeightsVis.jpg')

print('DONE')