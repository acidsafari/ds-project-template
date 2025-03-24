"""
Outlier detection and removal module for sensor data.

This module provides functions to detect and remove outliers from accelerometer
and gyroscope data collected during exercise movements. It implements various
outlier detection methods suitable for time-series sensor data.

Methods implemented:
- Statistical methods (z-score, IQR)
- Moving window statistics
- Exercise-specific boundary detection
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Set up file paths
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim")
INPUT_FILE = "01_sensor_data_resampled.pkl"
OUTPUT_FILE = "02_data_without_outliers.pkl"

# Load the sensor data
data = pd.read_pickle(os.path.join(INTERIM_DATA_PATH, INPUT_FILE))

outlier_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

# Params
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 100

# Single column test (like in tutorial comment)
# data[['acc_x', 'exercise_name']].boxplot(by='exercise_name', figsize=(20,10))

# data[outlier_columns[:3] + ['exercise_name']].boxplot(
#     by='exercise_name', figsize=(20,10), layout=(1,3)
# )

# data[outlier_columns[3:] + ['exercise_name']].boxplot(
#     by='exercise_name', figsize=(20,10), layout=(1,3)
# )

""" We are going to check the outliers over time
because some of them are going to be pretty normal
and should not be removed.
We are going to use a custom function for this.
"""

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. 
        Here, the col specifies the real data column and 
        outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


"""Before we use the above function, we need to create the outlier columns.
For that, we are going to use a different function, where to mark
the outliers using IQR (BELOW).
"""

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column
# col = "acc_x"
# dataset = mark_outliers_iqr(data, col)

# plot_binary_outliers(dataset=dataset, 
#                      col=col, 
#                      outlier_col=col + "_outlier", 
#                      reset_index=True
#                     )

# loop over all columns
# for col in data.select_dtypes(include=["int", "float"]).columns:
#     dataset = mark_outliers_iqr(data, col)
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col=col + "_outlier", 
#                          reset_index=True
#                         )

"""Looking at the values of the outliers, we can see that
we need to differenciate between exercise types, as some of them
are underrepresented and thus showing as outliers.
It looks like some of them are showing periods of rest, but
we need a model that is able to differenciate between them.
"""


# --------------------------------------------------------------
# Chauvenet's criterion (distribution based)
"""According to Chauvenet's criterion we reject a measurement 
(outlier) from a dataset of size N when it's probability of 
observation is less than 1/2N. 
A generalization is to replace the value 2 with a parameter C.

This assumes a normal distribution. So we need to check for that.
"""
# --------------------------------------------------------------

# Chauvenet's criterion function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds 
    a binary column with the same name extended with '_outlier' that 
    expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers 
                            given the assumption of a normal distribution, 
                            typically between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    
    dataset[col + "_outlier"] = mask
    
    return dataset

# Check for normal distribution
# data[outlier_columns[:3] + ['exercise_name']].plot.hist(
#     by='exercise_name', figsize=(20,20), layout=(3,3)
# )

# data[outlier_columns[3:] + ['exercise_name']].plot.hist(
#     by='exercise_name', figsize=(20,20), layout=(3,3)
# )

# loop over all columns
# for col in data.select_dtypes(include=["int", "float"]).columns:
#     dataset = mark_outliers_chauvenet(data, col)
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col=col + "_outlier", 
#                          reset_index=True
#                         )

"""From this we can see that there are a lot less outliers, and
that we still have a problem with the rest periods.

Before we choose the best method to deal with the outliers, we
will check LOF (local outlier factor).
"""


# --------------------------------------------------------------
# Local outlier factor (distance based)
"""The anomaly score of each sample is called the Local Outlier Factor. 
It measures the local deviation of the density of a given sample with 
respect to its neighbors. It is local in that the anomaly score depends 
on how isolated the object is with respect to the surrounding neighborhood. 
More precisely, locality is given by k-nearest neighbors, whose distance 
is used to estimate the local density. By comparing the local density 
of a sample to the local densities of its neighbors, one can identify 
samples that have a substantially lower density than their neighbors. 
These are considered outliers.
It requires training a model and making predictions.
KEY POINT: It compares full rows with neighborings rows, 
unlike previous methods.
"""
# --------------------------------------------------------------

# Inserting LOF function
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    
    return dataset, outliers, X_scores

# loop over all columns
# dataset, outliers, X_scores = mark_outliers_lof(dataset=data, columns=outlier_columns)
# for col in outlier_columns:
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col="outlier_lof", 
#                          reset_index=True
#                         )

"""We can definitely see a difference in the detection of outliers.
We are getting some that are in the middle of the distribution, plus
a more refined detection on previous edge cases, more closely to 
each exercise.
"""


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

# exercise_name = 'bench'
# for col in outlier_columns:
#     dataset = mark_outliers_iqr(data[data['exercise_name'] == exercise_name], col)
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col=col + "_outlier", 
#                          reset_index=True
#                         )

# for col in outlier_columns:
#     dataset = mark_outliers_chauvenet(data[data['exercise_name'] == exercise_name], col)
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col=col + "_outlier", 
#                          reset_index=True
#                         )

# dataset, outliers, X_scores = mark_outliers_lof(data[data['exercise_name'] == exercise_name], outlier_columns)
# for col in outlier_columns:
#     plot_binary_outliers(dataset=dataset, 
#                          col=col, 
#                          outlier_col="outlier_lof", 
#                          reset_index=True
#                         )

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# # Test on a single column
# col = 'gyro_x'
# dataset = mark_outliers_chauvenet(data, col=col)
# dataset[dataset[col + "_outlier"]]

# # Replace outliers with NaN for the test column
# dataset.loc[dataset[col + "_outlier"], col] = np.nan

# Function for full implementation (to be reviewed)
def replace_outliers_with_nan(dataset, columns, C=2):
    """Replace outliers with NaN in specified columns using Chauvenet's criterion.
    Outliers are detected separately for each exercise type.
    
    Args:
        dataset (pd.DataFrame): The dataset to process
        columns (list): List of column names to check for outliers
        C (int, optional): Parameter for Chauvenet's criterion. Defaults to 2.
    
    Returns:
        pd.DataFrame: A new dataframe with outliers replaced by NaN
    """
    # Create a copy to avoid modifying the original dataframe
    data_without_outliers = dataset.copy()
    
    # Get unique exercise names
    exercise_names = dataset['exercise_name'].unique()
    
    # Process each column
    for col in columns:
        # Process each exercise type for this column
        for exercise in exercise_names:
            # Get data for this exercise
            exercise_data = dataset[dataset['exercise_name'] == exercise].copy()
            
            if len(exercise_data) > 0:  # Only process if we have data
                # Mark outliers using Chauvenet's criterion
                marked_data = mark_outliers_chauvenet(exercise_data, col, C)
                
                # Replace values where outlier is True with NaN
                marked_data.loc[marked_data[col + "_outlier"], col] = np.nan
                
                # Update the values in the original dataframe
                data_without_outliers.loc[marked_data.index, col] = marked_data[col]
    
    return data_without_outliers

# data_without_outliers.info()


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

def export_data(data: pd.DataFrame, filename: str) -> None:
    """Export DataFrame to a pickle file in the interim data directory.
    
    Args:
        data (pd.DataFrame): The DataFrame to export
        filename (str): Name of the output file
    """
    output_path = os.path.join(INTERIM_DATA_PATH, filename)
    data.to_pickle(output_path)
    print(f"\nData exported to: {output_path}")
    print(f"Shape: {data.shape}")

# Test the export function with a single column
# col = 'gyro_x'
# dataset = mark_outliers_chauvenet(data, col=col)
# dataset.loc[dataset[col + "_outlier"], col] = np.nan
# export_data(dataset, OUTPUT_FILE)

# Full implementation (commented out for review)
# data_without_outliers = replace_outliers_with_nan(data, outlier_columns)
# export_data(data_without_outliers, OUTPUT_FILE)


# # Full implementation (commented out for review)
# data_without_outliers = replace_outliers_with_nan(data, outlier_columns)

# # Print summary of NaN values to see how many outliers were removed
# print("\nNumber of outliers removed per column:")
# print(data_without_outliers[outlier_columns].isna().sum())

# Print percentage of data points removed
# total_points = len(data) * len(outlier_columns)
# removed_points = data_without_outliers[outlier_columns].isna().sum().sum()
# print(f"\nPercentage of data points removed: {(removed_points/total_points)*100:.2f}%")
