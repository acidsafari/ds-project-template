"""
Development process for sensor data visualizations.

This file documents the iterative development of visualization functions for the sensor data analysis.
It shows the progression from basic plots to more sophisticated visualizations, including:

1. Initial data exploration with single column plots
2. Comparison of exercises and participants
3. Style adjustments and plot configurations
4. Development of the final dual sensor plot functionality

Note: This is a development file kept for reference. For production use, see visualize.py
"""

# --------------------------------------------------------------
# Imports and Setup
# --------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Initial Data Loading and Setup
# --------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
data = pd.read_pickle(os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl"))

# --------------------------------------------------------------
# Phase 1: Basic Data Exploration
# --------------------------------------------------------------

# Start with simple single column visualization to understand the data shape
set_df = data[data["set"] == 1]
plt.plot(set_df["acc_x"])

# Initial attempt at gyroscope data
# plt.plot(set_df["gyro_x"]).reset_index(drop=True)

# --------------------------------------------------------------
# Phase 2: Exercise Comparison
# --------------------------------------------------------------

# Plot all exercises to see patterns
for label in data["exercise_name"].unique():
    subset = data[data["exercise_name"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["gyro_x"].reset_index(drop=True), 
            label=label)
    plt.legend()
    plt.show()

# Zoom in to first 100 samples for more detail
for label in data["exercise_name"].unique():
    subset = data[data["exercise_name"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["gyro_x"].reset_index(drop=True), 
            label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Phase 3: Plot Configuration
# --------------------------------------------------------------

# Improve plot aesthetics with seaborn style
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Phase 4: Exercise Category Analysis
# --------------------------------------------------------------

# Compare medium vs heavy sets for squats
category_df = data.query("exercise_name == 'squat'").query("participant == 'A'").reset_index() 

# Group plot to show category differences
category_df.groupby(["exercise_category"])["acc_y"].plot()

fig, ax = plt.subplots()
category_df.groupby(["exercise_category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
ax.set_title("Squat Exercise - Heavy vs. Medium")
plt.legend()

"""
Key Finding: More acceleration observed with medium load as expected
"""

# --------------------------------------------------------------
# Phase 5: Participant Comparison
# --------------------------------------------------------------

# Compare different participants' bench press form
participant_df = data.query("exercise_name == 'bench'").sort_values(by="participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
ax.set_title("Bench Exercise - Participant Comparison")
plt.legend()

"""
Key Finding: Different exercise patterns visible for each participant
"""

# --------------------------------------------------------------
# Phase 6: Multi-Axis Development
# --------------------------------------------------------------

# Initial multi-axis plot development
exercise_name = "squat"
participant = "A"
all_axis_df = data.query(f"exercise_name == '{exercise_name}'").query(f"participant == '{participant}'").reset_index()

# Plot all axes to see movement in different dimensions
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc")
ax.set_xlabel("samples")
ax.set_title("Multi Axis - Exercise - Participant Comparison")
plt.legend()

"""
Key Finding: Different movement patterns visible in each axis
"""

# --------------------------------------------------------------
# Phase 7: Final Function Development
# --------------------------------------------------------------

def plot_sensor_participant_combinations(data: pd.DataFrame) -> None:
    """Initial version of combination plotting, later evolved into plot_dual_sensor_data"""
    # Function implementation...

def plot_dual_sensor_data(data: pd.DataFrame) -> None:
    """
    Create dual plots (accelerometer and gyroscope) for each exercise-participant combination.
    Plots accelerometer data in top row and gyroscope data in bottom row.
    
    Args:
        data (pd.DataFrame): The sensor data DataFrame
    """
    # Get unique combinations that have data
    combinations = []
    for exercise in data['exercise_name'].unique():
        for participant in data['participant'].unique():
            subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'")
            if not subset.empty:
                combinations.append((exercise, participant))
    
    # Create plots for each combination
    for exercise, participant in combinations:
        # Get data subset
        subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'").reset_index()
        
        if not subset.empty:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            
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
            
            # Create figures directory if it doesn't exist
            figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Save figure
            filename = f"{exercise}_({participant})_sensor_data.png"
            plt.savefig(os.path.join(figures_dir, filename), 
                       bbox_inches='tight', 
                       dpi=300)
            
            plt.show()

# Example usage:
# plot_dual_sensor_data(data)

# --------------------------------------------------------------
# Development Notes
# --------------------------------------------------------------
"""
Key Development Decisions:
1. Started with single sensor exploration
2. Added exercise and participant comparisons
3. Improved plot styling and configuration
4. Developed multi-axis visualization
5. Created final dual sensor plot function
6. Added high-resolution figure export

Future Improvements:
- Consider adding interactive plots
- Add options for different time windows
- Include statistical summaries
"""
