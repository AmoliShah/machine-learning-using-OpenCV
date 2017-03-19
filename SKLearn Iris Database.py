from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Y = iris.target

print list(iris.target_names)
print X
print Y
print X.shape
print Y.shape
print X_train.shape
print X_test.shape