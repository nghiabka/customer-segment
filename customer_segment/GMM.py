from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import customer_segment.preprocessing_data as pre
import  customer_segment.renders as rs


reduced_data, pca_samples, outliers = pre.preprocessing()
RAN_STATE = 29
FIG_SIZE = (14,8)


def score_GMM(n_clusters):
    # TODO: Apply your clustering algorithm of choice to the reduced data
    clusterer = GaussianMixture(n_components=n_clusters, random_state=RAN_STATE).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric="mahalanobis")
    return score

# Make an array of different silhouette_scores for clusters from 2 to 10.
GMM_scores = []
for i in range(2 ,12):
    GMM_scores.append(score_GMM(i))

# Plot silhouette scores for a variety of cluster numbers.
plt.figure(figsize= FIG_SIZE)
plt.plot(range(2 ,12) ,GMM_scores)
plt.xlabel('Amount of Clusters')
plt.ylabel('BIC scores')
plt.title('Plot of cluster number vs. silhouette score')
plt.show()
print ('Optimum score: {:.4f}'.format(min(GMM_scores)))

clusterer = GaussianMixture(n_components=2, random_state = RAN_STATE).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.means_
sample_preds = clusterer.predict(pca_samples)

rs.cluster_results(reduced_data, preds, centers, pca_samples)
