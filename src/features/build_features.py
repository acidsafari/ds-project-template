"""
Feature engineering for sensor data.
This module handles the creation of temporal and frequency features,
as well as clustering of the sensor data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
import os
from typing import Optional
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


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
    data.index.name = 'epoch (ms)'
    
    return data


def interpolate_missing_values(data: pd.DataFrame, 
                             columns: list = None, 
                             method: str = 'linear') -> pd.DataFrame:
    """Interpolate missing values in the specified columns.
    
    Args:
        data (pd.DataFrame): The sensor data
        columns (list, optional): List of columns to interpolate. If None, uses all predictor columns
            (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z). Defaults to None.
        method (str, optional): Interpolation method. Defaults to 'linear'.
    
    Returns:
        pd.DataFrame: DataFrame with interpolated values
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Create a copy to avoid modifying the original data
    data_clean = data.copy()
    
    # Get unique combinations of participant, exercise, and set
    groups = data_clean.groupby(['participant', 'exercise_name', 'set'])
    
    # Interpolate within each group
    for _, group in groups:
        indices = group.index
        data_clean.loc[indices, columns] = (
            data_clean.loc[indices, columns].interpolate(method=method)
        )
    
    return data_clean


def calculate_set_durations(data: pd.DataFrame, 
                          round_to_seconds: bool = True) -> pd.DataFrame:
    """Calculate the duration of each exercise set.
    
    Args:
        data (pd.DataFrame): The sensor data with datetime index
        round_to_seconds (bool, optional): Whether to round durations to whole seconds.
            Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame with added 'duration' column in seconds
    """
    # Create a copy to avoid modifying the original data
    data_with_duration = data.copy()
    
    # Calculate durations for each set
    groups = data_with_duration.groupby(['participant', 'exercise_name', 'set'])
    durations = []
    
    for _, group in groups:
        duration = (group.index[-1] - group.index[0]) / 1000  # Convert ms to seconds
        if round_to_seconds:
            duration = round(duration)
        durations.extend([duration] * len(group))
    
    data_with_duration['duration'] = durations
    return data_with_duration


def apply_butterworth_filter(data: pd.DataFrame,
                           columns: list = None, 
                           sampling_freq: float = 5.0,  # 1000/200 Hz
                           cutoff_freq: float = 1.3,    # Optimized value
                           order: int = 5) -> pd.DataFrame:
    """Apply Butterworth lowpass filter to the specified columns.
    
    Args:
        data (pd.DataFrame): The sensor data
        columns (list, optional): List of columns to filter. If None, uses all predictor columns.
            Defaults to None.
        sampling_freq (float, optional): Sampling frequency in Hz. Defaults to 5.0.
        cutoff_freq (float, optional): Cutoff frequency in Hz. Defaults to 1.3.
        order (int, optional): Filter order. Defaults to 5.
    
    Returns:
        pd.DataFrame: DataFrame with filtered columns
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Create a copy to avoid modifying the original data
    df_lowpass = data.copy()
    LowPass = LowPassFilter()
    
    # Apply filter to each column
    for col in columns:
        df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling_freq, cutoff_freq, order=order)
        df_lowpass[col] = df_lowpass[col + '_lowpass']
        del df_lowpass[col + '_lowpass']
    
    return df_lowpass


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


def save_clustered_data(data: pd.DataFrame,
                       filename: str = '03_data_features.pkl',
                       data_dir: str = None) -> None:
    """Save the clustered DataFrame to a pickle file.
    
    Args:
        data (pd.DataFrame): The DataFrame with cluster assignments
        filename (str, optional): Name of the output file.
            Defaults to '03_data_features.pkl'.
        data_dir (str, optional): Directory to save the file.
            If None, saves to data/interim/. Defaults to None.
    """
    if data_dir is None:
        data_dir = os.path.join('..', '..', 'data', 'interim')
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save DataFrame
    output_path = os.path.join(data_dir, filename)
    data.to_pickle(output_path)
    print(f"Saved clustered data to: {output_path}")


# --------------------------------------------------------------
# Visualization functions
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


def main():
    """Main function to run the feature engineering pipeline."""
    # Load and prepare data
    data = load_cleaned_sensor_data()
    
    # Interpolate missing values
    data_interpolated = interpolate_missing_values(data)
    
    # Calculate set durations
    data_with_duration = calculate_set_durations(data_interpolated)
    
    # Apply Butterworth filter
    df_lowpass = apply_butterworth_filter(data_with_duration)
    
    # Create clusters
    df_with_clusters = create_clustered_dataframe(df_lowpass)
    
    # Save the final dataset
    save_clustered_data(df_with_clusters)


if __name__ == "__main__":
    main()
