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


# Checking data loading
data = load_cleaned_sensor_data()

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

# Development code for visualization and testing
# Visualize the signal
# data[data['set'] == 25]['acc_y'].plot()
# data[data['set'] == 50]['acc_y'].plot()

# Single set duration
# duration = data_interpolated[data_interpolated['set'] == 1].index[-1] - data_interpolated[data_interpolated['set'] == 1].index[0]
# duration = duration.seconds

# # All sets duration
# for s in data_interpolated['set'].unique():
#     start = data_interpolated[data_interpolated['set'] == s].index[0]
#     stop = data_interpolated[data_interpolated['set'] == s].index[-1]
    
#     duration = stop - start
#     duration = duration.seconds
#     data_interpolated.loc[data_interpolated['set'] == s, 'duration'] = duration

# duration_df = data_interpolated.groupby('exercise_category')['duration'].mean()

# # Calculate average reps per set
# duration_df.iloc[0] / 5  # First category: 5 reps per set
# duration_df.iloc[1] / 10 # Second category: 10 reps per set

# Production-ready functions (to be moved to final version)
def calculate_set_durations(data: pd.DataFrame, round_to_seconds: bool = True) -> pd.DataFrame:
    """Calculate the duration of each exercise set and add it as a new column.
    
    Args:
        data (pd.DataFrame): The sensor data with datetime index
        round_to_seconds (bool, optional): Whether to round durations to whole seconds.
            Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame with added 'duration' column in seconds
    """
    data_with_duration = data.copy()
    
    for s in data['set'].unique():
        set_data = data[data['set'] == s]
        start = set_data.index[0]
        stop = set_data.index[-1]
        
        duration = (stop - start).total_seconds()
        if round_to_seconds:
            duration = round(duration)
        
        data_with_duration.loc[data['set'] == s, 'duration'] = duration
    
    return data_with_duration

data_with_duration = calculate_set_durations(data_interpolated)

# Calculate average durations
def get_average_durations(data: pd.DataFrame) -> pd.Series:
    """Calculate average durations for each exercise category.
    
    Args:
        data (pd.DataFrame): The sensor data with 'duration' and 'exercise_category' columns
    
    Returns:
        pd.Series: Average duration for each exercise category, rounded to 3 decimal places
    """
    return data.groupby('exercise_category')['duration'].mean().round(3)

average_durations_df = get_average_durations(data_with_duration)


# Test duration calculation if running as main script
# if __name__ == "__main__":
#     print("\n5. Testing set duration calculation:")
#     try:
#         # Load and prepare data
#         data = load_cleaned_sensor_data()
#         data_clean = interpolate_missing_values(data)
        
#         # Calculate durations
#         data_with_duration = calculate_set_durations(data_clean)
        
#         # Show some statistics
#         print("\nDuration statistics (seconds):")
#         print(f"Mean duration: {data_with_duration['duration'].mean():.1f}")
#         print(f"Min duration: {data_with_duration['duration'].min():.1f}")
#         print(f"Max duration: {data_with_duration['duration'].max():.1f}")
        
#         # Show average durations by exercise category
#         avg_durations = get_average_durations(data_with_duration)
#         print("\nAverage durations by exercise category:")
#         print(avg_durations)
        
#     except Exception as e:
#         print(f"✗ Error during duration calculation test: {str(e)}")


# --------------------------------------------------------------
# Butterworth lowpass filter
# NO NaN VALUES ALLOWED
# --------------------------------------------------------------

# Starting the class
df_lowpass = data_with_duration.copy()
LowPass = LowPassFilter()

# # Setting parameters
# fs = 1000 / 200 # the sampling frequency we set in section 2
# cutoff = 1.3 # this was optimised in the thesis project
# """we can play with the cutoff feature in the future, to see the
# effect it has in the model.
# """

# # Checking for one set and column
# df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff, order=5)

# subset = df_lowpass[df_lowpass['set'] == 45]
# print(subset["exercise_name"][0])

# # Visualizing the comparison
# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
# ax[0].plot(subset['acc_y'].reset_index(drop=True), 
#            label='raw_data')
# ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), 
#             label='butterworth_filter')
# ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#              fancybox=True, shadow=True)
# ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             fancybox=True, shadow=True)

# # Instead of adding a new column, we can replace the old column
# for col in predictor_columns:
#     df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
#     df_lowpass[col] = df_lowpass[col + '_lowpass']
#     del df_lowpass[col + '_lowpass']

# Production-ready function (to be moved to final version)
def apply_butterworth_filter(data: pd.DataFrame, 
                           columns: list = None, 
                           sampling_freq: float = 5.0,  # 1000/200 Hz
                           cutoff_freq: float = 1.3,    # Optimized value
                           order: int = 5) -> pd.DataFrame:
    """Apply Butterworth lowpass filter to the specified columns.
    
    Args:
        data (pd.DataFrame): The sensor data
        columns (list, optional): List of columns to filter. If None, uses all predictor columns.
        sampling_freq (float, optional): Sampling frequency in Hz. Defaults to 5.0 (1000/200).
        cutoff_freq (float, optional): Cutoff frequency in Hz. Defaults to 1.3.
        order (int, optional): Order of the Butterworth filter. Defaults to 5.
    
    Returns:
        pd.DataFrame: DataFrame with filtered sensor data
    """
    if columns is None:
        columns = predictor_columns
    
    # Create a copy to avoid modifying the original data
    filtered_data = data.copy()
    
    # Apply lowpass filter to each column
    for col in columns:
        filtered_data = LowPass.low_pass_filter(filtered_data, col, 
                                              sampling_freq, cutoff_freq, 
                                              order=order)
        # Replace original column with filtered data
        filtered_data[col] = filtered_data[col + '_lowpass']
        del filtered_data[col + '_lowpass']
    
    return filtered_data

# Apply the filter to our data
df_lowpass = apply_butterworth_filter(data_with_duration)

# subset = df_lowpass[df_lowpass['set'] == 45]
# print(subset["exercise_name"][0])

# # # Visualizing the comparison
# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
# ax[0].plot(subset['acc_y'].reset_index(drop=True), 
#            label='raw_data')
# ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), 
#             label='butterworth_filter')
# ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#              fancybox=True, shadow=True)
# ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             fancybox=True, shadow=True)


# --------------------------------------------------------------
# Principal component analysis PCA
""" We are going to use the functions in DataTransformation.py
for this.
"""
# --------------------------------------------------------------

# Development code
# Creating the class
# df_pca = df_lowpass.copy()
# pca = PrincipalComponentAnalysis()

# pc_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

# We are going to find the elbow value, optimal
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(predictor_columns) + 1), pc_values)
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Explained Variance')
# plt.title('Explained Variance by Principal Components')
# plt.show()

# Using 3 components
# df_pca = pca.apply_pca(df_pca, predictor_columns, 3)

# We are going to keep the original columns for now
# Checking an example
# subset = df_pca[df_pca['set'] == 35]
# subset[['pca_1', 'pca_2', 'pca_3']].plot()

# Production-ready function (to be moved to final version)
def create_pca_dataframe(data: pd.DataFrame, 
                        columns: list = None, 
                        n_components: int = 3) -> pd.DataFrame:
    """Apply Principal Component Analysis to the sensor data.
    
    Args:
        data (pd.DataFrame): The sensor data to transform
        columns (list, optional): List of columns to use for PCA. 
            If None, uses all predictor columns. Defaults to None.
        n_components (int, optional): Number of principal components to keep.
            Defaults to 3.
    
    Returns:
        pd.DataFrame: DataFrame with added PCA columns ('pca_1', 'pca_2', etc.)
        while preserving original columns
    """
    if columns is None:
        columns = predictor_columns
    
    # Create a copy to avoid modifying the original data
    df_pca = data.copy()
    
    # Initialize PCA
    pca = PrincipalComponentAnalysis()
    
    # Calculate explained variance (useful for analysis)
    pc_values = pca.determine_pc_explained_variance(df_pca, columns)
    
    # Apply PCA transformation
    df_pca = pca.apply_pca(df_pca, columns, n_components)
    
    return df_pca

# Apply PCA to our filtered data
df_with_pca = create_pca_dataframe(df_lowpass)


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# Development code
# df_square = df_pca.copy()

# acc_r = df_square['acc_x']**2 + df_square['acc_y']**2 + df_square['acc_z']**2
# gyro_r = df_square['gyro_x']**2 + df_square['gyro_y']**2 + df_square['gyro_z']**2

# df_square['acc_r'] = np.sqrt(acc_r)
# df_square['gyro_r'] = np.sqrt(gyro_r)

# subset = df_square[df_square['set'] == 14]
# subset[['acc_r', 'gyro_r']].plot(subplots=True, figsize=(10, 6))

"""we will look at the impact of this in the future
the reason why it could be good to use is that it does not
take into consideration of the direction of the movement
"""

# Production-ready function (to be moved to final version)
def calculate_squared_magnitudes(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the squared magnitudes (Euclidean norm) for accelerometer and gyroscope data.
    
    This function computes the root sum of squares for both accelerometer and gyroscope
    measurements, which gives a direction-independent magnitude of the motion.
    
    Args:
        data (pd.DataFrame): The sensor data containing accelerometer and gyroscope measurements
    
    Returns:
        pd.DataFrame: DataFrame with added 'acc_r' and 'gyro_r' columns containing
        the magnitude of acceleration and angular velocity respectively
    """
    # Create a copy to avoid modifying the original data
    df_square = data.copy()
    
    # Calculate squared sums
    acc_r = (df_square['acc_x']**2 + 
             df_square['acc_y']**2 + 
             df_square['acc_z']**2)
    
    gyro_r = (df_square['gyro_x']**2 + 
              df_square['gyro_y']**2 + 
              df_square['gyro_z']**2)
    
    # Add root sum of squares columns
    df_square['acc_r'] = np.sqrt(acc_r)
    df_square['gyro_r'] = np.sqrt(gyro_r)
    
    return df_square

# Apply squared magnitudes calculation to our PCA data
df_with_squares = calculate_squared_magnitudes(df_with_pca)

# --------------------------------------------------------------
# Temporal abstraction
"""We basically are going to calculate the rolling avg
with a window size and input that as a feature, which we can
use with all kinds of statistical features.
"""
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
