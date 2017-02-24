import cPickle
import numpy as np
files = 'VGGCascade_Opt_'
outNeurons = [64,128,256,512]
for i in outNeurons:
	currentFile = files + str(int(i)) + '.txt'
	print(currentFile)
	history = cPickle.load(open(currentFile,'r'))
	for j in range(6):
		currentIter = 'iter' + str(int(j))
		print((currentIter + ': %.3f') % (np.max(history[currentIter]['accuracyTest'])))
