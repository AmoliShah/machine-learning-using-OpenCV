#Logistic Regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X = iris.data
Y = iris.target
logrec = LogisticRegression()
logrec.fit(X,Y)
X_new = [[3,4,2,1],[1,2,3,4]]
print logrec.predict(X_new)