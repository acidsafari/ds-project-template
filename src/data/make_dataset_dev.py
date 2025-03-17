# -*- coding: utf-8 -*-
"""
This module handles data processing for the barbell exercise tracking project.
Interactive development is supported - use Shift+Enter to run code blocks.
"""

import pandas as pd
from glob import glob
import os

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

def read_csv_file(file_path):
    """Read a CSV file from the data directory"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Example path structure (adjust according to your data location):
# Absolute path from project root
DATA_PATH = "data/raw/"  # This will be relative to where you run the notebook from

# You can also use this more robust path that works regardless of where you run from:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ABSOLUTE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Example usage with an actual file from your dataset:
EXAMPLE_FILE = 'A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv'

def list_data_files():
    """List all CSV files in the raw data directory"""
    return glob(os.path.join(ABSOLUTE_DATA_PATH, '*.csv'))

def load_example_data():
    """Load an example CSV file from the raw data directory"""
    file_path = os.path.join(ABSOLUTE_DATA_PATH, EXAMPLE_FILE)
    return read_csv_file(file_path)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

def extract_features_from_filename(file_path: str) -> dict:
    """
    Extract features from a MetaWear data file path.
    Example path: /path/to/data/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
    
    Args:
        file_path (str): Full path to the data file
    
    Returns:
        dict: Dictionary containing extracted features
    """
    # Extract just the filename from the path
    filename = os.path.basename(file_path)
    
    # Remove .csv extension and split by underscores
    parts = filename.replace('.csv', '').split('_')
    
    # Extract exercise info (e.g., 'A-bench-heavy')
    exercise_info = parts[0].split('-')
    participant = exercise_info[0]  # e.g., 'A'
    exercise_name = exercise_info[1]  # e.g., 'bench'
    exercise_category = exercise_info[2].rstrip('123')  # e.g., 'heavy'
    # Option here to gain more features by creating reps, sets, etc.
    
    # Extract sensor info
    sensor_type = parts[4]  # e.g., 'Accelerometer'
    
    return {
        'participant': participant,
        'exercise_name': exercise_name,
        'exercise_category': exercise_category,
        'sensor_type': sensor_type
    }

# # Test with our example file using full path
# example_path = os.path.join(ABSOLUTE_DATA_PATH, EXAMPLE_FILE)
# print(f"Full path: {example_path}")
# print(f"Extracted filename: {os.path.basename(example_path)}\n")

# features = extract_features_from_filename(example_path)

# # View the extracted features
# for key, value in features.items():
#     print(f"{key}: {value}")


# --------------------------------------------------------------
# Create individual dataframes
# --------------------------------------------------------------
def create_dataframe_from_file(file_path: str) -> pd.DataFrame:
    """
    Create a pandas DataFrame containing:
    - the data in the .csv file
    - the metadata extracted from the filename as new columns.
    
    Args:
        file_path (str): Full path to the data file
        
    Returns:
        pd.DataFrame: DataFrame containing data and metadata
    """
    # Read the CSV data
    df = read_csv_file(file_path)
    
    # Extract features from filename
    features = extract_features_from_filename(file_path)
    
    # Add features as new columns
    for feature_name, feature_value in features.items():
        df[feature_name] = feature_value
    
    return df

# Example usage:
# df = create_dataframe_from_file(os.path.join(ABSOLUTE_DATA_PATH, EXAMPLE_FILE))
# print("\nDataFrame with features:")
# print(df.head())
# print("\nAll columns:")
# print(df.columns.tolist())


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
def create_sensor_dataframes_from_files(filelist: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create two pandas DataFrames, one for accelerometer and one for gyroscope data,
    containing the data from all files with their respective metadata.
    A 'set' counter is added to each dataframe starting from 1.
    
    Args:
        filelist (list): List of file paths to process
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (accelerometer_df, gyroscope_df)
    """
    # Initialize empty dataframes for each sensor type
    acce_df = pd.DataFrame()
    gyro_df = pd.DataFrame()
    
    # Initialize set counters
    acce_set_counter = 1
    gyro_set_counter = 1
    
    # Process each file
    for file_path in filelist:
        # Create dataframe with features
        df = create_dataframe_from_file(file_path)
        
        # Add to respective dataframe based on sensor type
        if df['sensor_type'].iloc[0] == 'Accelerometer':
            df['set'] = acce_set_counter
            acce_df = pd.concat([acce_df, df], ignore_index=True)
            acce_set_counter += 1
        elif df['sensor_type'].iloc[0] == 'Gyroscope':
            df['set'] = gyro_set_counter
            gyro_df = pd.concat([gyro_df, df], ignore_index=True)
            gyro_set_counter += 1
    
    return acce_df, gyro_df


# # TESTING THE FUNCTIONS
# # Get list of files
# data_files = glob(os.path.join(ABSOLUTE_DATA_PATH, "*.csv"))
# print(f"Found {len(data_files)} files")

# # Create sensor dataframes
# acc_df, gyro_df = create_sensor_dataframes_from_files(data_files)

# print("\nAccelerometer DataFrame:")
# print(f"- Shape: {acc_df.shape}")
# print(f"- Number of unique sets: {acc_df['set'].nunique()}")
# print(f"- Columns: {acc_df.columns.tolist()}")
# print("\nFirst few rows:")
# print(acc_df.head())

# print("\nGyroscope DataFrame:")
# print(f"- Shape: {gyro_df.shape}")
# print(f"- Number of unique sets: {gyro_df['set'].nunique()}")
# print(f"- Columns: {gyro_df.columns.tolist()}")
# print("\nFirst few rows:")
# print(gyro_df.head())

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

def process_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert epoch (ms) column to datetime and set as index.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'epoch (ms)' column
        
    Returns:
        pd.DataFrame: DataFrame with datetime index
    """
    # Convert epoch to datetime
    df.index = pd.to_datetime(df['epoch (ms)'], unit='ms')
    
    # Remove index name
    df.index.name = None
    
    return df

def clean_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove datetime-related columns after setting the index.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: DataFrame without datetime columns
    """
    # List of columns to drop
    datetime_cols = ['epoch (ms)', 'time (01:00)', 'elapsed (s)']
    
    # Drop columns if they exist
    cols_to_drop = [col for col in datetime_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


# #Example usage:
# acc_df = process_datetime_index(acc_df)
# acc_df = clean_datetime_columns(acc_df)
# print("\nDataFrame with datetime index:")
# print(acc_df.head())
# print("\nColumns after cleaning:")
# print(acc_df.columns.tolist())

# # Get list of files and create sensor dataframes
# data_files = glob(os.path.join(ABSOLUTE_DATA_PATH, "*.csv"))
# acc_df, gyro_df = create_sensor_dataframes_from_files(data_files)

# # Process accelerometer data
# print("Processing accelerometer data:")
# print("Before processing:")
# print(acc_df.head())

# acc_df = process_datetime_index(acc_df)
# acc_df = clean_datetime_columns(acc_df)

# print("\nAfter processing:")
# print(acc_df.head())
# print("\nColumns after cleaning:")
# print(acc_df.columns.tolist())

# # Process gyroscope data
# print("\nProcessing gyroscope data:")
# print("Before processing:")
# print(gyro_df.head())

# gyro_df = process_datetime_index(gyro_df)
# gyro_df = clean_datetime_columns(gyro_df)

# print("\nAfter processing:")
# print(gyro_df.head())
# print("\nColumns after cleaning:")
# print(gyro_df.columns.tolist())

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

def merge_sensor_dataframes(acc_df: pd.DataFrame, gyro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge accelerometer and gyroscope dataframes based on their datetime index.
    Takes only the first three columns from accelerometer data and all from gyroscope.
    
    Args:
        acc_df (pd.DataFrame): Accelerometer DataFrame with datetime index
        gyro_df (pd.DataFrame): Gyroscope DataFrame with datetime index
        
    Returns:
        pd.DataFrame: Merged DataFrame containing both sensor data
    """
    # Select only first three columns from accelerometer (excluding metadata)
    acc_sensor_cols = [col for col in acc_df.columns 
                      if col not in ['participant', 'exercise_name', 'exercise_category', 'set', 'sensor_type']][:3]
    acc_df_subset = acc_df[acc_sensor_cols]
    
    # Combine metadata with sensor data
    merged_df = pd.concat([acc_df_subset, gyro_df], axis=1).drop(columns=['sensor_type'])
    
    # Rename columns
    merged_df.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'participant', 'exercise_name', 'exercise_category', 'set']
    
    return merged_df

# # Example usage:
# # Process and merge sensor dataframes
# # Process and merge sensor dataframes
# data_files = glob(os.path.join(ABSOLUTE_DATA_PATH, "*.csv"))
# acc_df, gyro_df = create_sensor_dataframes_from_files(data_files)

# # Process datetime for both dataframes
# acc_df = process_datetime_index(acc_df)
# acc_df = clean_datetime_columns(acc_df)
# gyro_df = process_datetime_index(gyro_df)
# gyro_df = clean_datetime_columns(gyro_df)

# # Merge dataframes
# merged_df = merge_sensor_dataframes(acc_df, gyro_df)

# print("Merged DataFrame info:")
# print(merged_df.info())
# print("\nFirst few rows:")
# print(merged_df.head())
# print("\nColumns:")
# print(merged_df.columns.tolist())

# Process and merge sensor dataframes
data_files = glob(os.path.join(ABSOLUTE_DATA_PATH, "*.csv"))
acc_df, gyro_df = create_sensor_dataframes_from_files(data_files)

# Process datetime for both dataframes
acc_df = process_datetime_index(acc_df)
acc_df = clean_datetime_columns(acc_df)
gyro_df = process_datetime_index(gyro_df)
gyro_df = clean_datetime_columns(gyro_df)

# Merge dataframes
merged_df = merge_sensor_dataframes(acc_df, gyro_df)

# Resample data
days = [g for n, g in merged_df.groupby(pd.Grouper(freq='D'))]
resampled_df = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])
resampled_df["set"] = resampled_df["set"].astype(int)

# Save to pickle
save_to_pickle(resampled_df, "sensor_data_resampled.pkl")

# # Verify we can load it back
# test_df = pd.read_pickle(os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl"))
# print("\nLoaded DataFrame info:")
# print(test_df.info())
# print("\nFirst few rows:")
# print(test_df.head())
# print("\nColumns in order:")
# print(test_df.columns.tolist())

# if __name__ == '__main__':
#     # This code will run when executing the file directly
#     # but won't run when importing the file    files = list_data_files()
#     print("Available data files:", files)

def create_processed_dataset() -> pd.DataFrame:
    """
    Create the processed sensor dataset from raw files.
    
    Returns:
        pd.DataFrame: Processed and resampled dataset
    """
    # Get list of files and create sensor dataframes
    data_files = glob(os.path.join(ABSOLUTE_DATA_PATH, "*.csv"))
    acc_df, gyro_df = create_sensor_dataframes_from_files(data_files)
    
    # Process datetime for both dataframes
    acc_df = process_datetime_index(acc_df)
    acc_df = clean_datetime_columns(acc_df)
    gyro_df = process_datetime_index(gyro_df)
    gyro_df = clean_datetime_columns(gyro_df)
    
    # Merge dataframes
    merged_df = merge_sensor_dataframes(acc_df, gyro_df)
    
    # Resample data
    days = [g for n, g in merged_df.groupby(pd.Grouper(freq='D'))]
    resampled_df = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])
    resampled_df["set"] = resampled_df["set"].astype(int)
    
    return resampled_df

if __name__ == '__main__':
    # Create and save the processed dataset
    print("Creating processed dataset...")
    processed_df = create_processed_dataset()
    
    print("Saving dataset to pickle file...")
    save_to_pickle(processed_df, "sensor_data_resampled.pkl")
    
    print("Done!")

# For interactive development (Shift+Enter):
# Example usage:
# processed_df = create_processed_dataset()
# save_to_pickle(processed_df, "sensor_data_resampled.pkl")
#
# # Load back and verify
# test_df = pd.read_pickle(os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl"))
# print(test_df.info())

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

def save_to_pickle(df: pd.DataFrame, filename: str) -> None:
    """
    Save DataFrame to a pickle file in the interim data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the file (without path)
    """
    # Create the interim data path
    interim_path = os.path.join(PROJECT_ROOT, "data", "interim")
    
    # Create the directory if it doesn't exist
    os.makedirs(interim_path, exist_ok=True)
    
    # Full path for the pickle file
    pickle_path = os.path.join(interim_path, filename)
    
    # Save DataFrame to pickle
    df.to_pickle(pickle_path)
    print(f"DataFrame saved to: {pickle_path}")

# Example usage:
# save_to_pickle(resampled_df, "sensor_data_resampled.pkl")

# --------------------------------------------------------------
# Resample data (frequency conversion)
# Accelerometer:    12.500HZ 8/s
# Gyroscope:        25.000HZ 4/s
# --------------------------------------------------------------

# Resample accelerometer and gyroscope data
sampling = {
    'acc_x': 'mean',
    'acc_y': 'mean', 
    'acc_z': 'mean', 
    'gyro_x': 'mean', 
    'gyro_y': 'mean', 
    'gyro_z': 'mean', 
    'participant': 'last', 
    'exercise_name': 'last', 
    'exercise_category': 'last', 
    'set': 'last'
}

# This results in a large df
# resampled_df = merged_df.resample(rule='200ms').apply(sampling)

# Split by day - trick to reduce the memory usage
days = [g for n, g in merged_df.groupby(pd.Grouper(freq='D'))]

resampled_df = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

# resample set with style fix
resampled_df["set"] = resampled_df["set"].astype(int)

# Testing results
print("Resampled DataFrame info:")
print(resampled_df.info())
print("\nFirst few rows:")
print(resampled_df.head())
print("\nColumns in order:")
print(resampled_df.columns.tolist())
