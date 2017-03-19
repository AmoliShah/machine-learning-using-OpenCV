# Clustering Iris database with K Nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Y = iris.target
knn = KNeighborsClassifier()
print knn
knn.fit(X,Y)
print knn.predict ([3,5,4,2])
