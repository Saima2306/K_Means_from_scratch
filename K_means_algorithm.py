import random
import numpy as np
class  KMeans:
    def __init__(self,n_clusters = 2,max_iterations= 100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
    
    def fit_predict(self,X):
        rand_index = random.sample(range(0,X.shape[0]),self.n_clusters) #selecting the index from the X data
        self.centroids = X[rand_index]
        # print(self.centroids)
        # assign cluster
        for i in range(self.max_iterations):
            # assign the cluster
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            #move centroids
            self.centroids = self.move_centroids(X,cluster_group)
            # check the last statement
            if (old_centroids == self.centroids).all():
                break
        return cluster_group
            
    def assign_clusters(self,X):
        cluster_group = []
        # calculate the distance of each data point to each centroid(100 points and 2 centroids, calculate 200 distance
        # and ask the point which is nearer to you, if 0 wala centroid then 0 else 1)
        distances = []
        for row in X:
            for centroid in self.centroids:
                #general formula of eucledian distance
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            min_index_position = distances.index(min_distance)
            cluster_group.append(min_index_position)
            distances.clear()
        return np.array(cluster_group)
    #select random centroids which is equal to number of clusters created
    def move_centroids(self,X,cluster_group):
        """
        in cluster group, we have 0 and 1's now if we want to move the centroids we need to shift to calculate the average of 0's and 
        # 1's class """
        new_centroids = []
        # we have to extract the unique clusters in our program
        unique_clusters = np.unique(cluster_group) 
        for type in unique_clusters:
            new_centroids = X[cluster_group == type].mean(axis=0)
        return np.array(new_centroids)
