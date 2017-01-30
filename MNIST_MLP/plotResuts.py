import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

history = cPickle.load(open('mlp2layers.txt','r'))
accuracyTestCascade = np.hstack((history['iter0']['accuracyTest'],history['iter1']['accuracyTest']))
plt.plot(accuracyTestCascade,lw=2,alpha=3,label='Cascade Training')
plt.hold(True)
plt.plot(history['normalTraining']['accuracyTest'],lw=2,alpha=3,label='End-to-End')
plt.legend(loc=0,prop={'size' : 12})
plt.xlabel('EPOCHS')
plt.xlim((0,100))
plt.ylabel('ACCURACY')
#plt.ylim([0.5,1.05])
plt.savefig('./accuracyTest.jpg',bbox_inches='tight',format='jpg',dmi=1000)