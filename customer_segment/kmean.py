from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np


import customer_segment.preprocessing_data as pre
import  customer_segment.renders as rs


reduced_data, pca_samples, outliers = pre.preprocessing()
RAN_STATE = 29
FIG_SIZE = (14,8)




# wcss = []
# for i in range(2,12):
#     km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
#     km.fit(reduced_data)
#     wcss.append(km.inertia_)
# plt.plot(range(2,12),wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss')
# plt.show()


# k means determine k
distortions = []
K = range(2,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(reduced_data)
    kmeanModel.fit(reduced_data)
    distortions.append(sum(np.min(cdist(reduced_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / reduced_data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



# cluster
clusterer = KMeans(n_clusters=5, random_state = RAN_STATE).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.cluster_centers_
sample_preds = clusterer.predict(pca_samples)
rs.cluster_results(reduced_data, preds, centers, pca_samples)
