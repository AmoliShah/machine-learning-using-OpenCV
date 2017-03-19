# Creating boundary on Gaussian NB
import numpy as np
import matplotlib as plt
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])  #collection of features
Y = np.array([1, 1, 1, 2, 2, 2])  #class or label to which feature belongs to (responses)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()  #instance of Gaussian NB model
clf.fit(X, Y)   #training of model
x = [-1,-2,-3,1,2,3]
y = [-1,-1,-2,1,1,2]
h=.02






#create a mesh to plot in
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min, y_max = X[:,1].min() - 1,X[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.coolwarm)
plt.show()



