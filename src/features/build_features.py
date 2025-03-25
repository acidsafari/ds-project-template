import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
import os
from typing import Optional


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

def load_cleaned_sensor_data(version: str = 'default') -> pd.DataFrame:
    """Load the cleaned sensor data from the interim data directory.
    
    Args:
        version (str, optional): Which version of cleaned data to load. Options:
            - 'default': Standard cleaned data (Chauvenet's criterion)
            - 'iqr': IQR-based outlier removal
            - 'chauvenet': Chauvenet's criterion
            - 'lof': Local Outlier Factor
            Default is 'default'.
    
    Returns:
        pd.DataFrame: The cleaned sensor data
    
    Raises:
        FileNotFoundError: If the specified data file doesn't exist
        ValueError: If an invalid version is specified
    """
    # Set up file paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim")
    
    # Map version to filename
    version_map = {
        'default': "02_data_without_outliers.pkl",
        'iqr': "02a_data_without_outliers_iqr.pkl",
        'chauvenet': "02b_data_without_outliers_chauvenet.pkl",
        'lof': "02c_data_without_outliers_lof.pkl"
    }
    
    if version not in version_map:
        raise ValueError(f"Invalid version '{version}'. Must be one of: {list(version_map.keys())}")
    
    file_path = os.path.join(INTERIM_DATA_PATH, version_map[version])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load and return the data
    return pd.read_pickle(file_path)


# # Checking data loading
# data = load_cleaned_sensor_data()

# Define predictor columns (sensor data columns)
predictor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Plot settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# We are going to use interpolation for this
# --------------------------------------------------------------

def interpolate_missing_values(data: pd.DataFrame, columns: list = None, method: str = 'linear') -> pd.DataFrame:
    """Interpolate missing values in the specified columns using pandas interpolation.
    
    Args:
        data (pd.DataFrame): The sensor data
        columns (list, optional): List of columns to interpolate. If None, uses all predictor columns
            (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z). Defaults to None.
        method (str, optional): Interpolation method. Options:
            - 'linear': Linear interpolation
            - 'cubic': Cubic spline interpolation
            - 'nearest': Nearest neighbor interpolation
            - 'quadratic': Quadratic interpolation
            Defaults to 'linear'.
    
    Returns:
        pd.DataFrame: DataFrame with interpolated values
    
    Note:
        The function interpolates values separately for each exercise set to avoid
        interpolating across different movements.
    """
    if columns is None:
        columns = predictor_columns
    
    # Create a copy to avoid modifying the original data
    data_clean = data.copy()
    
    # Get unique combinations of participant, exercise, and set
    groups = data_clean.groupby(['participant', 'exercise_name', 'set'])
    
    # Interpolate within each group
    for _, group in groups:
        indices = group.index
        data_clean.loc[indices, columns] = (
            group[columns]
            .interpolate(method=method, limit_direction='both')
        )
    
    # Check if any NaN values remain
    remaining_nans = data_clean[columns].isna().sum()
    if remaining_nans.any():
        print("Warning: Some NaN values could not be interpolated:")
        print(remaining_nans[remaining_nans > 0])
    
    return data_clean

data_interpolated = interpolate_missing_values(data)

# data_interpolated.info()

# # Test interpolation if running as main script
# if __name__ == "__main__":
#     print("\n4. Testing interpolation:")
#     try:
#         # Load data and show initial NaN counts
#         data = load_cleaned_sensor_data(version='iqr')  # Using IQR version as it has most NaNs
#         print("\nBefore interpolation:")
#         print(data[predictor_columns].isna().sum())
        
#         # Interpolate and show results
#         data_clean = interpolate_missing_values(data)
#         print("\nAfter interpolation:")
#         print(data_clean[predictor_columns].isna().sum())
        
#         # Show percentage of values that were interpolated
#         total_values = len(data) * len(predictor_columns)
#         interpolated = (data[predictor_columns].isna().sum().sum())
#         print(f"\nPercentage of values interpolated: {(interpolated/total_values)*100:.2f}%")
        
#     except Exception as e:
#         print(f"✗ Error during interpolation test: {str(e)}")


# --------------------------------------------------------------
# Calculating set duration
"""We want to calculate the set duration and the rep duration, so that
we can use them as variables for the lowpass filter. In this way,
we can eliminate the noise from the signal. 

This is a complex calculation, for which we are going to use the
functions we imported from the ML4QS course.
"""
# --------------------------------------------------------------

# Visualize the signal
# data[data['set'] == 25]['acc_y'].plot()
# data[data['set'] == 50]['acc_y'].plot()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# if __name__ == "__main__":
#     print("Testing data loading function...")
    
#     # Test default version
#     print("\n1. Loading default version (Chauvenet):")
#     try:
#         data = load_cleaned_sensor_data()
#         print(f"✓ Successfully loaded data with shape: {data.shape}")
#         print(f"✓ Columns: {', '.join(data.columns)}")
#         print(f"✓ Number of exercises: {len(data['exercise_name'].unique())}")
#     except Exception as e:
#         print(f"✗ Error loading default data: {str(e)}")
    
#     # Test all versions
#     versions = ['iqr', 'chauvenet', 'lof']
#     for version in versions:
#         print(f"\n2. Loading {version} version:")
#         try:
#             data = load_cleaned_sensor_data(version=version)
#             print(f"✓ Successfully loaded data with shape: {data.shape}")
#             print(f"✓ NaN values per column:")
#             print(data.isna().sum())
#         except Exception as e:
#             print(f"✗ Error loading {version} data: {str(e)}")
    
#     # Test error handling
#     print("\n3. Testing error handling:")
#     try:
#         data = load_cleaned_sensor_data(version='invalid')
#         print("✗ Should have raised ValueError")
#     except ValueError as e:
#         print(f"✓ Correctly caught invalid version: {str(e)}")
