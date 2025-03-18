import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import os

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data = pd.read_pickle(os.path.join(PROJECT_ROOT, "data", "interim", "sensor_data_resampled.pkl"))

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = data[data["set"] == 1]
plt.plot(set_df["acc_x"])

# plt.plot(set_df["gyro_x"]).reset_index(drop=True)

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in data["exercise_name"].unique():
    subset = data[data["exercise_name"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["gyro_x"].reset_index(drop=True), 
            label=label)
    plt.legend()
    plt.show()

# To gain a little more definiton
for label in data["exercise_name"].unique():
    subset = data[data["exercise_name"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["gyro_x"].reset_index(drop=True), 
            label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings - with rcParams
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
# generating the subset
category_df = data.query("exercise_name == 'squat'").query("participant == 'A'").reset_index() 

# group plot
category_df.groupby(["exercise_category"])["acc_y"].plot()

fig, ax = plt.subplots()
category_df.groupby(["exercise_category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
ax.set_title("Squat Exercise - Heavy vs. Medium")
plt.legend()

"""
The plot shows how there is more acceleration with the medium load
as expected
"""

# --------------------------------------------------------------
# Compare participants
# necessary to generalise the models
# --------------------------------------------------------------

participant_df = data.query("exercise_name == 'bench'").sort_values(by="participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
ax.set_title("Bench Exercise - Participant Comparison")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

exercise_name = "squat"
participant = "A"
all_axis_df = data.query(f"exercise_name == '{exercise_name}'").query(f"participant == '{participant}'").reset_index()

# plot
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc")
ax.set_xlabel("samples")
ax.set_title("Multi Axis - Exercise - Participant Comparison")
plt.legend()

"""
We can see the different exercises patterns for each person
"""

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

def plot_sensor_participant_combinations(data: pd.DataFrame) -> None:
    """
    Plot all combinations of exercise_name and participant for a given sensor type.
    
    Args:
        data (pd.DataFrame): The sensor data DataFrame
        sensor_type (str): Either 'acc' for accelerometer or 'gyro' for gyroscope
    """
    # Get sensor columns
    sensor_cols = [col for col in data.columns if col.startswith('acc') or col.startswith('gyro')]
    
    # Get unique combinations that have data
    combinations = []
    for exercise in data['exercise_name'].unique():
        for participant in data['participant'].unique():
            subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'")
            if not subset.empty:
                combinations.append((exercise, participant))
    
    # Create plots
    for exercise, participant in combinations:
        # Get data subset
        subset = data.query(f"exercise_name == '{exercise}' and participant == '{participant}'").reset_index()
        
        if not subset.empty:
            # Create plot
            fig, ax = plt.subplots()
            subset[sensor_cols].plot(ax=ax)
            
            # Set labels and title
            ax.set_ylabel("sensor reading")
            ax.set_xlabel("samples")
            ax.set_title(f"{exercise} ({participant})".title())
            plt.legend()
            

# Example usage:
# Plot accelerometer data
plot_sensor_participant_combinations(data)


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------