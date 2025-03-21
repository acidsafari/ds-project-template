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

# Load the sensor data
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl")
data = pd.read_pickle(DATA_PATH)

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
col = "acc_x"
dataset = mark_outliers_iqr(data, col)

plot_binary_outliers(dataset=dataset, 
                     col=col, 
                     outlier_col=col + "_outlier", 
                     reset_index=True
                    )

# loop over all columns
for col in data.select_dtypes(include=["int", "float"]).columns:
    dataset = mark_outliers_iqr(data, col)
    plot_binary_outliers(dataset=dataset, 
                         col=col, 
                         outlier_col=col + "_outlier", 
                         reset_index=True
                        )

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


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
