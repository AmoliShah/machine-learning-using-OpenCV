#Creating Boundary on SVM Classifier (Plotting)
import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm
X = np.array([[0,0],[1,1],[2,2],[-1,-1,]])
Y = np.array([1,2,3,4])
clf = svm.SVC()
clf.fit(X, Y)
print(clf.predict([[-.08,-1]]))

x=[0,1,2,-1]
y=[0,1,2,-1]

x_min= X[:,0].min()-1
x_max= X[:,0].max() + 1
y_min= X[:,1].min()-1
y_max= X[:,1].max() + 1

h=.02

plt.scatter(X[:,0],X[:,1],s=100,c='red',alpha=0.5)

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])

Z= Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=.8)

plt.scatter(X[:,0],X[:,1],c='yellow',cmap=plt.cm.coolwarm)

plt.show()