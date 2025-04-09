import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
import os
from typing import Optional
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
    
    # Load the data
    data = pd.read_pickle(file_path)
    
    # Set the index name
    data.index.name = 'epoch (ms)'
    
    return data


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
We create this using the NumericalAbstraction class.
"""
# --------------------------------------------------------------

# # Development code
# df_temporal = df_with_squares.copy()
# num_abstraction = NumericalAbstraction()

# predictor_columns = predictor_columns + ['acc_r', 'gyro_r']

# window_size = int(1000 / 200) # 5 seconds

# # This will introduce errors in the data, mixing different exercises
# for col in predictor_columns:
#     df_temporal = num_abstraction.abstract_numerical(df_temporal, [col], window_size, 'mean')
#     df_temporal = num_abstraction.abstract_numerical(df_temporal, [col], window_size, 'std')
    
# # To calculate for each set
# df_temporal_list = []

# for set in df_temporal['set'].unique():
#     subset = df_temporal[df_temporal['set'] == set].copy()
#     for col in predictor_columns:
#         subset = num_abstraction.abstract_numerical(subset, [col], window_size, 'mean')
#         subset = num_abstraction.abstract_numerical(subset, [col], window_size, 'std')
#     df_temporal_list.append(subset)

# # Combine all sets back into a single DataFrame
# df_temporal = pd.concat(df_temporal_list)
# df_temporal.info()

# subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()
# subset[['gyro_y', 'gyro_y_temp_mean_ws_5', 'gyro_y_temp_std_ws_5']].plot()


# Production-ready function
def create_temporal_features(data: pd.DataFrame, 
                           columns: list = predictor_columns,
                           window_size: int = 5,
                           aggregation_functions: list = None) -> pd.DataFrame:
    """Create temporal features for sensor data using rolling windows.
    
    The function calculates temporal features separately for each exercise set to avoid
    mixing data between different exercises. Features are calculated using a rolling
    window approach.
    
    Args:
        data (pd.DataFrame): The sensor data to process
        columns (list, optional): List of columns to create features for.
            If None, uses predictor columns + magnitude columns. Defaults to None.
        window_size (int, optional): Size of the rolling window. Defaults to 5.
        aggregation_functions (list, optional): List of aggregation functions to apply.
            Available options: ['mean', 'max', 'min', 'median', 'std'].
            Defaults to ['mean', 'std'].
    
    Returns:
        pd.DataFrame: DataFrame with added temporal features for each column:
            - {col}_temp_{function}_ws_{window_size}: Rolling aggregation
    """
    # Create a copy of the columns list to avoid modifying the original
    if columns is predictor_columns:
        columns = predictor_columns.copy()
        if 'acc_r' in data.columns and 'gyro_r' in data.columns:
            columns.extend(['acc_r', 'gyro_r'])
    
    # Initialize abstraction class
    num_abstraction = NumericalAbstraction()
    
    # Process each set separately to avoid mixing exercise data
    df_temporal_list = []
    
    for set_id in data['set'].unique():
        # Get data for this set
        subset = data[data['set'] == set_id].copy()
        
        # Set default aggregation functions if none provided
        if aggregation_functions is None:
            aggregation_functions = ['mean', 'std']
            
        # Calculate temporal features for each column
        for col in columns:
            for agg_func in aggregation_functions:
                subset = num_abstraction.abstract_numerical(subset, [col], window_size, agg_func)
        
        df_temporal_list.append(subset)
    
    # Combine all sets while preserving the index
    df_temporal = pd.concat(df_temporal_list, axis=0)
    
    # Sort by the original index to maintain time series order
    df_temporal = df_temporal.sort_index()
    
    return df_temporal

# Apply temporal feature creation to our squared magnitudes data
df_with_temporal = create_temporal_features(df_with_squares)

# df_with_temporal.info()

# # Visualise results
# subset = df_with_temporal[df_with_temporal['set'] == 15]
# subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()
# subset[['gyro_y', 'gyro_y_temp_mean_ws_5', 'gyro_y_temp_std_ws_5']].plot()


# --------------------------------------------------------------
# Frequency features
"""We are going to use the Discrete Fourier Transform (DFT) to calculate 
the frequency features.
https://en.wikipedia.org/wiki/Discrete_Fourier_transform
We will abstract the different frequency components.
This is all based on the original GitHub repo ML4QS.
Features we will be extracting:
• Amplitude (for each of the relevant frequencies that are part of the time window)
• Max frequency
• Weighted frequency (average)
• Power spectral entropy
"""
# --------------------------------------------------------------

# # Development code
# df_frequency = df_with_temporal.copy().reset_index()
# freq_abstraction = FourierTransformation()

# # parameters
# freq_size = int(1000 / 200) # 5 readings per second
# window_size = int(2800 / 200) # 14 readings per avg set

# # Apply frequency abstraction
# df_frequency = freq_abstraction.abstract_frequency(df_frequency, ['acc_y'], window_size, freq_size)

# # Visualise results
# subset = df_frequency[df_frequency['set'] == 15]
# subset[['acc_y']].plot()
# subset[['acc_y_max_freq',
#         'acc_y_freq_weighted',
#         'acc_y_pse',
#         'acc_y_freq_1.429_Hz_ws_14', 
#         'acc_y_freq_2.5_Hz_ws_14', 
#         ]
# ].plot()

# # To calculate for each set
# df_frequency_list = []

# for set in df_frequency['set'].unique():
#     print(f"Processing set {set}")
#     subset = df_frequency[df_frequency['set'] == set].reset_index(drop=True).copy()
#     subset = freq_abstraction.abstract_frequency(subset, predictor_columns, window_size, freq_size)
#     df_frequency_list.append(subset)

# # Combine all sets back into a single DataFrame
# # we have to put back the index, after the discrete one we created above
# df_frequency = pd.concat(df_frequency_list).set_index('epoch (ms)', drop=True)


# Production-ready function
def create_frequency_features(data: pd.DataFrame,
                            columns: list = None,
                            window_size: int = 14,  # 14 readings per avg set
                            freq_size: int = 5,    # 5 readings per second
                            verbose: bool = False) -> pd.DataFrame:
    """Create frequency-based features using Discrete Fourier Transform (DFT).
    
    This function calculates various frequency-based features for each exercise set:
    • Amplitude for relevant frequencies in the time window
    • Maximum frequency
    • Weighted frequency (average)
    • Power spectral entropy
    
    Args:
        data (pd.DataFrame): The sensor data to process
        columns (list, optional): List of columns to create features for.
            If None, uses predictor columns. Defaults to None.
        window_size (int, optional): Size of the rolling window in data points.
            Defaults to 14 (average set length).
        freq_size (int, optional): Number of frequency components to consider.
            Defaults to 5.
        verbose (bool, optional): Whether to print progress messages.
            Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame with added frequency-based features for each column:
            - {col}_max_freq: Maximum frequency
            - {col}_freq_weighted: Weighted average frequency
            - {col}_pse: Power spectral entropy
            - {col}_freq_{freq}_Hz_ws_{window}: Amplitude at specific frequencies
    """
    if columns is None:
        columns = predictor_columns.copy()
        columns.extend(['acc_r', 'gyro_r'])
    
    # Initialize frequency abstraction
    freq_abstraction = FourierTransformation()
    
    # Reset index temporarily for frequency calculations
    df_frequency = data.copy()
    index_name = df_frequency.index.name
    df_frequency = df_frequency.reset_index()
    
    # Process each set separately
    df_frequency_list = []
    
    for set_id in df_frequency['set'].unique():
        if verbose:
            print(f"Processing set {set_id}")
        
        # Get data for this set
        subset = df_frequency[df_frequency['set'] == set_id].reset_index(drop=True).copy()
        
        # Calculate frequency features
        subset = freq_abstraction.abstract_frequency(subset, columns, window_size, freq_size)
        
        df_frequency_list.append(subset)
    
    # Combine all sets and restore the index
    df_frequency = pd.concat(df_frequency_list)
    df_frequency = df_frequency.set_index(index_name)
    
    # Sort by the original index to maintain time series order
    df_frequency = df_frequency.sort_index()
    
    return df_frequency

# Apply frequency feature creation to our temporal data
df_with_frequency = create_frequency_features(df_with_temporal)


# --------------------------------------------------------------
# Dealing with overlapping windows
"""We are going to drop the NaN values resulting from the overlapping windows.
This is a common issue when working with time series data and overlapping windows.

In order to get rid off the overlaping windows, we are going to drop 50% of the values, even though
it sounds counterintuitive, it is a common practice in time series analysis.
By doing this, we are reducing the autocorrelation that occurs between adjacent rows.
It has been proven that this is a valid approach, in order
to make the models less prone to overfitting.
"""
# --------------------------------------------------------------

# # Development code
# df_no_nan = df_with_frequency.dropna()

# df_no_overlap = df_no_nan.iloc[::2]


def remove_overlapping_windows(data: pd.DataFrame,
                             sampling_rate: int = 2,
                             drop_nan: bool = True) -> pd.DataFrame:
    """Remove overlapping windows from time series data.
    
    This function handles the common issue of overlapping windows in time series data.
    It first removes NaN values (optional) and then reduces autocorrelation by
    keeping only every nth sample (where n is the sampling_rate).
    
    This is a common practice in time series analysis to reduce overfitting by
    decreasing the autocorrelation between adjacent rows.
    
    Args:
        data (pd.DataFrame): The input DataFrame with potential overlapping windows
        sampling_rate (int, optional): Keep every nth sample. Defaults to 2 (50% reduction).
        drop_nan (bool, optional): Whether to drop NaN values before sampling.
            Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame with reduced overlapping windows
    """
    # First handle NaN values if requested
    df_processed = data.dropna() if drop_nan else data.copy()
    
    # Remove overlapping windows by taking every nth sample
    df_processed = df_processed.iloc[::sampling_rate]
    
    return df_processed


# Apply overlap removal to our frequency features
df_final = remove_overlapping_windows(df_with_frequency)


# --------------------------------------------------------------
# Clustering
"""UNSUPERVISED LEARNING
we are goig to calculate the k-means with the elbow method.
"""
# --------------------------------------------------------------

# Development code
df_cluster = df_final.copy()

# Calculate k-means with elbow method
cluster_cols = ['acc_x', 'acc_y', 'acc_z']
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_cols]
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Model for k = 5
kmeans = KMeans(n_clusters=5, random_state=0)
subset = df_cluster[cluster_cols]
df_cluster['cluster'] = kmeans.fit_predict(subset)

# Plot and save clusters visualization
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
for cluster in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == cluster]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=f'Cluster {cluster}')
ax.set_xlabel('Acceleration X (g)')
ax.set_ylabel('Acceleration Y (g)')
ax.set_zlabel('Acceleration Z (g)')
ax.set_title('K-Means Clustering Results (k=5)', pad=20, fontsize=14)
ax.legend()

# Save clusters plot
plot_dir = os.path.join('..', '..', 'reports', 'development_presentation', 'images')
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, 'clusters_plot.png'),
            bbox_inches='tight', dpi=300)
plt.show()

# Plot and save accelerometer data comparison
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
for exercise_name in df_cluster['exercise_name'].unique():
    subset = df_cluster[df_cluster['exercise_name'] == exercise_name]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=f'Exercise {exercise_name}')
ax.set_xlabel('Acceleration X (g)')
ax.set_ylabel('Acceleration Y (g)')
ax.set_zlabel('Acceleration Z (g)')
ax.set_title('Accelerometer Data by Exercise Type', pad=20, fontsize=14)
ax.legend()

# Save exercise comparison plot
plot_dir = os.path.join('..', '..', 'reports', 'development_presentation', 'images')
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, 'exercise_comparison_plot.png'),
            bbox_inches='tight', dpi=300)
plt.show()




# --------------------------------------------------------------
# Clustering functions
# --------------------------------------------------------------

def create_clustered_dataframe(data: pd.DataFrame,
                             n_clusters: int = 5,
                             columns: list = None,
                             random_state: int = 0) -> pd.DataFrame:
    """Create a DataFrame with cluster assignments for each data point.
    
    Args:
        data (pd.DataFrame): The data to cluster
        n_clusters (int, optional): Number of clusters for K-means.
            Defaults to 5.
        columns (list, optional): Columns to use for clustering.
            Defaults to ['acc_x', 'acc_y', 'acc_z'].
        random_state (int, optional): Random state for reproducibility.
            Defaults to 0.
    
    Returns:
        pd.DataFrame: Original DataFrame with an additional 'cluster' column
            containing cluster assignments (0 to n_clusters-1)
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z']
    
    # Create clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    subset = data[columns]
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = kmeans.fit_predict(subset)
    
    return data_with_clusters


# Example usage of clustering function
df_with_clusters = create_clustered_dataframe(df_final, n_clusters=5)


# --------------------------------------------------------------
# Clustering visualization functions
# --------------------------------------------------------------

def visualize_elbow_method(data: pd.DataFrame,
                        columns: list = None,
                        k_range: tuple = (2, 10),
                        random_state: int = 0,
                        save_plot: bool = False,
                        save_dir: str = None) -> None:
    """Visualize the elbow method for K-means clustering to help choose optimal k.
    
    Args:
        data (pd.DataFrame): The data to cluster
        columns (list, optional): Columns to use for clustering.
            Defaults to ['acc_x', 'acc_y', 'acc_z'].
        k_range (tuple, optional): Range of k values to test (min, max).
            Defaults to (2, 10).
        random_state (int, optional): Random state for reproducibility.
            Defaults to 0.
        save_plot (bool, optional): Whether to save the plot.
            Defaults to False.
        save_dir (str, optional): Directory to save the plot.
            Required if save_plot is True.
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z']
    
    k_values = range(k_range[0], k_range[1])
    inertias = []
    
    for k in k_values:
        subset = data[columns]
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(subset)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 10))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal k', pad=20, fontsize=14)
    
    if save_plot:
        if save_dir is None:
            raise ValueError("save_dir must be provided when save_plot is True")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'elbow_curve.png'),
                    bbox_inches='tight', dpi=300)
    
    plt.show()

# For elbow method visualization
visualize_elbow_method(df_with_clusters)


def plot_cluster_comparison(data: pd.DataFrame,
                           n_clusters: int = 5,
                           columns: list = None,
                           random_state: int = 0) -> None:
    """Plot K-means clustering results alongside actual exercise labels.
    
    Creates two 3D scatter plots side by side:
    1. Data points colored by cluster assignment
    2. Same data points colored by exercise type
    
    Args:
        data (pd.DataFrame): The data to cluster and visualize
        n_clusters (int, optional): Number of clusters for K-means.
            Defaults to 5.
        columns (list, optional): Columns to use for clustering.
            Defaults to ['acc_x', 'acc_y', 'acc_z'].
        random_state (int, optional): Random state for reproducibility.
            Defaults to 0.
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z']
    
    # Create clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    subset = data[columns]
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = kmeans.fit_predict(subset)
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 15))
    
    # Plot clusters
    ax1 = fig.add_subplot(121, projection='3d')
    for cluster in data_with_clusters['cluster'].unique():
        subset = data_with_clusters[data_with_clusters['cluster'] == cluster]
        ax1.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
                    label=f'Cluster {cluster}')
    ax1.set_xlabel('Acceleration X (g)')
    ax1.set_ylabel('Acceleration Y (g)')
    ax1.set_zlabel('Acceleration Z (g)')
    ax1.set_title(f'K-Means Clustering Results (k={n_clusters})',
                  pad=20, fontsize=14)
    ax1.legend()
    
    # Plot exercise types
    ax2 = fig.add_subplot(122, projection='3d')
    for exercise_name in data_with_clusters['exercise_name'].unique():
        subset = data_with_clusters[data_with_clusters['exercise_name'] == exercise_name]
        ax2.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
                    label=f'Exercise {exercise_name}')
    ax2.set_xlabel('Acceleration X (g)')
    ax2.set_ylabel('Acceleration Y (g)')
    ax2.set_zlabel('Acceleration Z (g)')
    ax2.set_title('Accelerometer Data by Exercise Type',
                  pad=20, fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# For cluster comparison (no saving)
plot_cluster_comparison(df_with_clusters, n_clusters=5)


def save_cluster_plot(data: pd.DataFrame,
                      n_clusters: int = 5,
                      columns: list = None,
                      random_state: int = 0,
                      save_dir: str = None) -> None:
    """Create and save a single plot showing K-means clustering results.
    
    Args:
        data (pd.DataFrame): The data to cluster and visualize
        n_clusters (int, optional): Number of clusters for K-means.
            Defaults to 5.
        columns (list, optional): Columns to use for clustering.
            Defaults to ['acc_x', 'acc_y', 'acc_z'].
        random_state (int, optional): Random state for reproducibility.
            Defaults to 0.
        save_dir (str): Directory to save the plot.
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z']
    
    if save_dir is None:
        raise ValueError("save_dir must be provided")
    
    # Create clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    subset = data[columns]
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = kmeans.fit_predict(subset)
    
    # Create single plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    # Plot clusters
    for cluster in data_with_clusters['cluster'].unique():
        subset = data_with_clusters[data_with_clusters['cluster'] == cluster]
        ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'],
                   label=f'Cluster {cluster}')
    
    ax.set_xlabel('Acceleration X (g)')
    ax.set_ylabel('Acceleration Y (g)')
    ax.set_zlabel('Acceleration Z (g)')
    ax.set_title(f'K-Means Clustering Results (k={n_clusters})',
                 pad=20, fontsize=14)
    ax.legend()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'cluster_exercise_side_by_side_plot.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# Example usage of visualization functions
plot_dir = os.path.join('..', '..', 'reports', 'development_presentation', 'images')
save_cluster_plot(df_with_clusters, n_clusters=5, save_dir=plot_dir)


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

def save_clustered_data(data: pd.DataFrame,
                       filename: str = 'clustered_sensor_data.pkl',
                       data_dir: str = None) -> None:
    """Save the clustered DataFrame to a pickle file.
    
    Args:
        data (pd.DataFrame): The DataFrame with cluster assignments
        filename (str, optional): Name of the output file.
            Defaults to 'clustered_sensor_data.pkl'.
        data_dir (str, optional): Directory to save the file.
            If None, saves to data/processed/. Defaults to None.
    """
    if data_dir is None:
        data_dir = os.path.join('..', '..', 'data', 'interim')
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save DataFrame
    output_path = os.path.join(data_dir, filename)
    data.to_pickle(output_path)
    print(f"Saved clustered data to: {output_path}")


# Save the clustered DataFrame to interim folder with standard naming
save_clustered_data(df_with_clusters, filename='03_data_features.pkl')


# Make it production ready.

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
