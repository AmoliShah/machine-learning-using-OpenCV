#Naive Bayes Classifier
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])  #collection of features
Y = np.array([1, 1, 1, 2, 2, 2])  #class or label to which feature belongs to (responses)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()  #instance of Gaussian NB model
clf.fit(X, Y)   #training of model
GaussianNB(priors=None)
print(clf.predict([[-0.8, -1]]))