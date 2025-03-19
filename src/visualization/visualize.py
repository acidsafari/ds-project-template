"""
Visualization module for sensor data analysis.

This module provides functions to create visualizations of accelerometer and gyroscope
data from exercise movements. The main functionality includes:

- Dual sensor plots (accelerometer and gyroscope) for each exercise-participant combination
- High-resolution figure export to reports/figures directory
- Configurable plot styling and layout

Example:
    To generate all visualizations:
    ```
    python src/visualization/visualize.py
    ```
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

import os
import pandas as pd
import matplotlib as mpl
# Use non-interactive backend
mpl.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Set up paths
# --------------------------------------------------------------

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Set plot style
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["figure.figsize"] = (20, 10)

# --------------------------------------------------------------
# Visualization functions
# --------------------------------------------------------------

def plot_dual_sensor_data(data: pd.DataFrame) -> None:
    """
    Create dual plots (accelerometer and gyroscope) for each exercise-participant combination.
    
    For each valid combination of exercise and participant, this function creates a figure
    with two subplots:
    - Top: Accelerometer data (x, y, z axes)
    - Bottom: Gyroscope data (x, y, z axes)
    
    The plots share an x-axis for time alignment and include clear labels and legends.
    Generated figures are saved as high-resolution PNGs in the reports/figures directory.
    
    Args:
        data (pd.DataFrame): DataFrame containing sensor data with columns:
            - exercise_name: Name of the exercise
            - participant: Participant identifier
            - acc_x, acc_y, acc_z: Accelerometer data
            - gyro_x, gyro_y, gyro_z: Gyroscope data
    """
    # Get unique combinations that have data
    combinations = []
    for exercise in data['exercise_name'].unique():
        for participant in data['participant'].unique():
            subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'")
            if not subset.empty:
                combinations.append((exercise, participant))
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Generating {len(combinations)} plots...")
    
    # Create plots for each combination
    for i, (exercise, participant) in enumerate(combinations, 1):
        # Get data subset
        subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'").reset_index()
        
        if not subset.empty:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            
            # Plot accelerometer data
            acc_cols = [col for col in subset.columns if col.startswith('acc')]
            subset[acc_cols].plot(ax=ax1)
            ax1.set_ylabel('Accelerometer')
            ax1.legend(labels=['acc_x', 'acc_y', 'acc_z'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.15),
                       ncol=3,
                       fancybox=True,
                       shadow=True)
            
            # Plot gyroscope data
            gyro_cols = [col for col in subset.columns if col.startswith('gyro')]
            subset[gyro_cols].plot(ax=ax2)
            ax2.set_ylabel('Gyroscope')
            ax2.set_xlabel('samples')
            ax2.legend(labels=['gyro_x', 'gyro_y', 'gyro_z'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.15),
                       ncol=3,
                       fancybox=True,
                       shadow=True)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Save figure
            filename = f"{exercise}_({participant})_sensor_data.png"
            plt.savefig(os.path.join(figures_dir, filename), 
                       bbox_inches='tight', 
                       dpi=300)
            
            # Close the figure to free memory
            plt.close(fig)
            
            print(f"Progress: {i}/{len(combinations)} - Generated plot for {exercise} ({participant})")

# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

if __name__ == "__main__":
    print("Generating sensor data visualizations...")
    
    # Load the data
    print("Loading sensor data...")
    data = pd.read_pickle(os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl"))
    
    # Create visualizations
    print("Creating dual sensor plots for all exercise-participant combinations...")
    plot_dual_sensor_data(data)
    
    print("Visualizations completed! Check the reports/figures directory for the generated plots.")
