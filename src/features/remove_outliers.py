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
data[['acc_x', 'exercise_name']].boxplot(by='exercise_name', figsize=(20,10))

data[outlier_columns[:3] + ['exercise_name']].boxplot(
    by='exercise_name', figsize=(20,10), layout=(1,3)
)

data[outlier_columns[3:] + ['exercise_name']].boxplot(
    by='exercise_name', figsize=(20,10), layout=(1,3)
)





# Create histograms for accelerometer
plt.figure(figsize=(15,5))
for idx, col in enumerate(outlier_columns[:3]):
    plt.subplot(1, 3, idx+1)
    plt.hist(data[col], bins=50)
    plt.title(f'{col}')
    plt.xlabel('Acceleration (g)')
plt.tight_layout()
plt.show()

# Create histograms for gyroscope
plt.figure(figsize=(15,5))
for idx, col in enumerate(outlier_columns[3:]):
    plt.subplot(1, 3, idx+1)
    plt.hist(data[col], bins=50)
    plt.title(f'{col}')
    plt.xlabel('Angular Velocity (deg/s)')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# --------------------------------------------------------------
# Chauvenet's criterion (distribution based)
# --------------------------------------------------------------

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
