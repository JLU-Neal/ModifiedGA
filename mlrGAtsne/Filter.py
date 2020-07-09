import numpy as np
import random
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
class Filter():
    def randomFilter(self,temp,y):
        origin_size=temp.shape[0]
        post_filter_size=int(origin_size/12)
        X=np.array([])
        Y=np.array([])
        for index in range(post_filter_size):
            if X.size==0:
                i=random.randint(0,origin_size-1)
                X=temp[i]
                Y=y[i]
            else:
                i=random.randint(0,origin_size-1)
                X=np.vstack((X,temp[i]))
                Y=np.vstack((Y,y[i]))
        return X,Y

    def APFilter(self,temp,Y):
        X=temp.copy()
        # Compute Affinity Propagation
        af = AffinityPropagation(damping=.5).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        cluster_center=np.array([])
        value=np.array([])
        n_clusters_ = len(cluster_centers_indices)
        for k in zip(range(n_clusters_)):
            if cluster_center.size==0:
                cluster_center = temp[cluster_centers_indices[k]]
                value=Y[cluster_centers_indices[k]]
            else:
                cluster_center=np.vstack((cluster_center,temp[cluster_centers_indices[k]]))
                value=np.vstack((value,Y[cluster_centers_indices[k]]))

        if n_clusters_<=10:
            cluster_center=temp
            value=Y
        print(n_clusters_)
        return cluster_center,value


