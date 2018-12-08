from sklearn.cluster import DBSCAN
import customer_segment.preprocessing_data as pre


reduced_data, pca_samples, outliers = pre.preprocessing()

db = DBSCAN(eps=0.3, min_samples=10).fit(reduced_data)
