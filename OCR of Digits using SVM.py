import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the data, converters convert the letter to a number
data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data,2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

svm_params = dict(kernel_type = cv2.SVM_RBF,C=100,gamma=0.01)

# Initiate the kNN, classify, measure accuracy.
svm=cv2.SVM()
svm.train(trainData,responses,params=svm_params)
result = svm.predict_all(testData)
correct= np.count_nonzero(result ==labels)

accuracy = correct*100.0/10000
print accuracy

from sklearn import svm
clf = svm.SVC()

x_min= trainData[:,0].min()-1
x_max= trainData[:,0].max() + 1
y_min= responses[:,0].min()-1
y_max= responses[:,0].max() + 1
h=0.02

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
clf.fit(trainData,responses) 
result =  clf.predict(np.c_[xx.ravel(),yy.ravel()])
correct= np.count_nonzero(result ==labels)
accuracy = correct*100.0/10000
print accuracy