#K Means Clustering and plotting
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)   # random state remembers the last value(cluster) assigned to the given data
print kmeans.labels_    # label = 0 or 1 as no of clusters = 2
print kmeans.predict([[0, 0], [4, 4]])
print kmeans.cluster_centers_
cent = kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1],c='blue',s=5)
plt.scatter(cent[:,0],cent[:,1],c='red',s=50)
plt.show()