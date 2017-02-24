from __future__ import print_function
import numpy as np
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

history = cPickle.load(open('./VGG_Results/VGGCascade_Opt_BiggerNet_Dropout.txt','r'))
timeSeriesCL_Test = list()
timeSeriesCL_Training = list()
for i in range(len(history.keys())):
	timeSeriesCL_Test.append(history['iter'+str(int(i))]['accuracyTest'])
	timeSeriesCL_Training.append(history['iter'+str(int(i))]['accuracyTraining'])

print(('%.4f')%(np.max(timeSeriesCL_Test)))
timeSeriesCL_Test = np.hstack(timeSeriesCL_Test)
timeSeriesCL_Training = np.hstack(timeSeriesCL_Training)

timeSeriesEE_Test = np.asarray(cPickle.load(open('./VGG_Results/End-End.txt','r'))['normalTraining']['accuracyTest'])
timeSeriesEE_Training = np.asarray(cPickle.load(open('./VGG_Results/End-End.txt','r'))['normalTraining']['accuracyTraining'])

plt.figure(1)
plt.plot(timeSeriesCL_Test,lw=2,alpha=3,label='Cascade Training')
plt.hold(True)
plt.plot(timeSeriesEE_Test,lw=2,alpha=3,label='End-to-End')
plt.legend(loc=0,prop={'size' : 12})
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY - TEST')
#plt.ylim([0.5,1.05])
plt.savefig('./VGG_Results/accuracyTest_BiggerNet.jpg',bbox_inches='tight',format='jpg',dmi=1000)

plt.figure(2)
plt.plot(timeSeriesCL_Training,lw=2,alpha=3,label='Cascade Training')
plt.hold(True)
plt.plot(timeSeriesEE_Training,lw=2,alpha=3,label='End-to-End')
plt.legend(loc=0,prop={'size' : 12})
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY - TRAINING')
#plt.ylim([0.5,1.05])
plt.savefig('./VGG_Results/accuracyTraining_BiggerNet.jpg',bbox_inches='tight',format='jpg',dmi=1000)

plt.figure(3)
plt.plot(timeSeriesCL_Test,lw=2,alpha=3,label='Testing')
plt.hold(True)
plt.plot(timeSeriesCL_Training,lw=2,alpha=3,label='Training')
plt.legend(loc=0,prop={'size' : 12})
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY - TRAINING')
#plt.ylim([0.5,1.05])
plt.savefig('./VGG_Results/accuracyTrainingvsTesting_BiggerNet.jpg',bbox_inches='tight',format='jpg',dmi=1000)


pass