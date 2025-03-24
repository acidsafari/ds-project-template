"""
Outlier detection and removal module for sensor data.

This module provides functions to detect and remove outliers from accelerometer
and gyroscope data collected during exercise movements using multiple methods:
- Interquartile Range (IQR)
- Chauvenet's criterion
- Local Outlier Factor (LOF)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os
import scipy
from sklearn.neighbors import LocalOutlierFactor

def mark_outliers_iqr(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mark outliers in the specified column using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset to process
        col (str): Name of the column to mark outliers in

    Returns:
        pd.DataFrame: Dataset with an additional boolean column indicating outliers
    """
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)

    return dataset

def mark_outliers_chauvenet(dataset: pd.DataFrame, col: str, C: float = 2) -> pd.DataFrame:
    """Mark outliers in the specified column using Chauvenet's criterion.
    
    Args:
        dataset (pd.DataFrame): The dataset to process
        col (str): Name of the column to mark outliers in
        C (float, optional): Parameter for Chauvenet's criterion. Defaults to 2.
    
    Returns:
        pd.DataFrame: Dataset with an additional boolean column indicating outliers
    """
    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Compute the probability of observing the data points
    prob = []
    for data in dataset[col]:
        # Compute the z-score and corresponding probability
        z = abs(data - mean)/std
        prob.append(2.0 * (1 - scipy.stats.norm.cdf(z)))

    # Mark as outliers where probability < criterion
    dataset[col + "_outlier"] = [p < criterion for p in prob]
    return dataset

def mark_outliers_lof(dataset: pd.DataFrame, columns: List[str], n: int = 20) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Mark outliers using Local Outlier Factor (LOF).
    
    Args:
        dataset (pd.DataFrame): The dataset to process
        columns (List[str]): List of columns to use for LOF
        n (int, optional): Number of neighbors. Defaults to 20.
    
    Returns:
        Tuple containing:
        - pd.DataFrame: Dataset with an additional boolean column 'outlier_lof'
        - np.ndarray: Binary outlier labels (-1 for outliers, 1 for inliers)
        - np.ndarray: Negative outlier factors
    """
    dataset = dataset.copy()
    
    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_
    
    dataset["outlier_lof"] = outliers == -1
    
    return dataset, outliers, X_scores

def replace_outliers_with_nan(dataset: pd.DataFrame, columns: List[str], method: str = 'chauvenet', **kwargs) -> pd.DataFrame:
    """Replace outliers with NaN in specified columns using the chosen method.
    Outliers are detected separately for each exercise type.
    
    Args:
        dataset (pd.DataFrame): The dataset to process
        columns (List[str]): List of column names to check for outliers
        method (str, optional): Outlier detection method ('iqr', 'chauvenet', or 'lof'). Defaults to 'chauvenet'.
        **kwargs: Additional arguments for the outlier detection method:
            - C (float): Parameter for Chauvenet's criterion (default: 2)
            - n_neighbors (int): Number of neighbors for LOF (default: 20)
    
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
                # Mark outliers using the specified method
                if method == 'iqr':
                    marked_data = mark_outliers_iqr(exercise_data, col)
                    outlier_col = col + "_outlier"
                elif method == 'chauvenet':
                    C = kwargs.get('C', 2)
                    marked_data = mark_outliers_chauvenet(exercise_data, col, C)
                    outlier_col = col + "_outlier"
                elif method == 'lof':
                    n_neighbors = kwargs.get('n_neighbors', 20)
                    marked_data, _, _ = mark_outliers_lof(exercise_data, [col], n_neighbors)
                    outlier_col = "outlier_lof"
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Replace values where outlier is True with NaN
                marked_data.loc[marked_data[outlier_col], col] = np.nan
                
                # Update the values in the original dataframe
                data_without_outliers.loc[marked_data.index, col] = marked_data[col]
    
    return data_without_outliers

def process_outliers(input_file: str, output_file: str, columns: List[str], method: str = 'chauvenet', **kwargs) -> None:
    """Process outliers in sensor data and save the cleaned dataset.
    
    Args:
        input_file (str): Path to input pickle file
        output_file (str): Path to save output pickle file
        columns (List[str]): List of columns to process for outliers
        method (str, optional): Outlier detection method ('iqr', 'chauvenet', or 'lof'). Defaults to 'chauvenet'.
        **kwargs: Additional arguments passed to replace_outliers_with_nan
    """
    # Load data
    data = pd.read_pickle(input_file)
    
    # Process outliers
    print(f"\nProcessing outliers using {method} method...")
    data_without_outliers = replace_outliers_with_nan(data, columns, method, **kwargs)
    
    # Print summary
    print("\nNumber of outliers removed per column:")
    print(data_without_outliers[columns].isna().sum())
    
    total_points = len(data) * len(columns)
    removed_points = data_without_outliers[columns].isna().sum().sum()
    print(f"\nPercentage of data points removed: {(removed_points/total_points)*100:.2f}%")
    
    # Save cleaned data
    data_without_outliers.to_pickle(output_file)
    print(f"\nData exported to: {output_file}")
    print(f"Shape: {data_without_outliers.shape}")

if __name__ == "__main__":
    # Set up file paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim")
    INPUT_FILE = os.path.join(INTERIM_DATA_PATH, "01_sensor_data_resampled.pkl")
    OUTPUT_FILE = os.path.join(INTERIM_DATA_PATH, "02_data_without_outliers.pkl")
    
    # Define columns to process
    OUTLIER_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Process outliers using default method (Chauvenet's criterion)
    process_outliers(INPUT_FILE, OUTPUT_FILE, OUTLIER_COLUMNS)

    # # Uncomment to test different outlier detection methods
    # print("\nTesting different outlier detection methods...")
    # 
    # # Test IQR method
    # print("\n1. IQR Method:")
    # process_outliers(
    #     INPUT_FILE, 
    #     os.path.join(INTERIM_DATA_PATH, "02a_data_without_outliers_iqr.pkl"),
    #     OUTLIER_COLUMNS,
    #     method='iqr'
    # )
    # 
    # # Test Chauvenet's criterion
    # print("\n2. Chauvenet's Criterion:")
    # process_outliers(
    #     INPUT_FILE,
    #     os.path.join(INTERIM_DATA_PATH, "02b_data_without_outliers_chauvenet.pkl"),
    #     OUTLIER_COLUMNS,
    #     method='chauvenet',
    #     C=2
    # )
    # 
    # # Test LOF method
    # print("\n3. Local Outlier Factor:")
    # process_outliers(
    #     INPUT_FILE,
    #     os.path.join(INTERIM_DATA_PATH, "02c_data_without_outliers_lof.pkl"),
    #     OUTLIER_COLUMNS,
    #     method='lof',
    #     n_neighbors=20
    # )
