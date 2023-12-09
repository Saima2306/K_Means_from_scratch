from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from K_means_algorithm import KMeans

centroids = [(-5,-5),(5,5)]
cluster_stds = [1,1]
X,y = make_blobs(n_samples=100,cluster_std=cluster_stds,centers=centroids,n_features=2,random_state=2)
km = KMeans(n_clusters=2,max_iterations=200)
y_means= km.fit_predict(X)
print(y_means)
plt.scatter(X[y_means==0,0],X[y_means==0,1],color = 'red')
plt.scatter(X[y_means==1,0],X[y_means==1,1],color = 'blue')

# plt.scatter(X[:,0],X[:,1])
plt.show()