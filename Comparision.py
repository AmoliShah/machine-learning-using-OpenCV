# Comparing KNN and Logistic Regression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time 
iris = load_iris()
X = iris.data
Y = iris.target


# Logistic Regression
logrec = LogisticRegression()
t0 = time.time()
logrec.fit(X,Y)
#print time.time() - t0     #Time taken to fit tha data into model
Y_lpred = logrec.predict(X)
#print len(Y_lpred)
#print Y
#print Y_pred
print "Logistic Regression Accuracy",metrics.accuracy_score(Y,Y_lpred)


# KNN = 5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,Y)
Y_kpred = knn.predict(X)
print "KNN = 5 Accuracy", metrics.accuracy_score(Y,Y_kpred)


# KNN = 1
knn1 = KNeighborsClassifier(n_neighbors = 1)
knn1.fit(X,Y)
Y_k1pred = knn1.predict(X)
print "KNN = 1 Accuracy" , metrics.accuracy_score(Y,Y_k1pred)