import pandas
import pandas as pd
import numpy as np
from IPython.display import display  # Allows the use of display() for DataFrames
from sklearn.decomposition import PCA

import customer_segment.renders as rs

RAN_STATE = 29
FIG_SIZE = (14, 8)

try:
    data = pd.read_csv("Wholesale customers data.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))

except:
    print("Dataset could not be loaded. Is the dataset missing?")


def outlierDetection(log_data):
    all_outliers = []
    # For each feature find the data points with extreme high or low values
    for feature in log_data.keys():

        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(log_data[feature], 25)

        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(log_data[feature], 75)

        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = (Q3 - Q1) * 1.5

        # Display the outliers
        print("Data points considered outliers for the feature '{}':".format(feature))
        outlier_points = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
        display(outlier_points)

        for index in outlier_points.index.values:
            all_outliers.append(index)

    # Find which data points are outliers on two or more features.
    all_outliers, indices = np.unique(all_outliers, return_inverse=True)
    counts = np.bincount(indices)
    outliers_2_or_more = all_outliers[counts > 1]
    print('Outliers on two or more features:', outliers_2_or_more)

    # Select the indices for data points you wish to remove
    outliers = outliers_2_or_more

    # Remove the outliers, if any were specified
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

    return good_data, outlier_points, outliers


def preprocessing():
    # TODO: Select three indices of your choice you wish to sample from the dataset
    indices = [181, 325, 86]

    # Create a DataFrame of the chosen samples
    samples = pandas.DataFrame(data.loc[indices], columns=data.columns)  # .reset_index(drop = True)
    # TODO: Scale the data using the natural logarithm
    log_data = np.log(data)

    # TODO: Scale the sample data using the natural logarithm
    log_samples = np.log(samples)

    good_data, outlier_points, outliers = outlierDetection(log_data)

    # Feature Transformation¶
    # TODO: Apply PCA to the good data with the same number of dimensions as features
    pca = PCA(n_components=6).fit(good_data)

    # TODO: Apply a PCA transformation to the sample log-data
    pca_samples = pca.transform(log_samples)

    # Generate PCA results plot
    # pca_results = rs.pca_results(good_data, pca)

    # TODO: Apply PCA by fitting the good data with only two dimensions
    pca = PCA(n_components=2).fit(good_data)

    # TODO: Transform the good data using the PCA fit above
    reduced_data = pca.transform(good_data)

    # TODO: Transform the sample log-data using the PCA fit above
    pca_samples = pca.transform(log_samples)

    # Create a DataFrame for the reduced data
    reduced_data = pandas.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])

    return reduced_data, pca_samples, outliers



indices = [181, 325, 86]


samples = pandas.DataFrame(data.loc[indices], columns=data.columns)  # .reset_index(drop = True)
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

good_data, outlier_points, outliers = outlierDetection(log_data)
