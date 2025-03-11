import pandas as pd
from glob import glob

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
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ABSOLUTE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# Example usage with an actual file from your dataset:
EXAMPLE_FILE = 'A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv'

def load_example_data():
    """Load an example accelerometer data file"""
    file_path = os.path.join(ABSOLUTE_DATA_PATH, EXAMPLE_FILE)
    return read_csv_file(file_path)

example_read = load_example_data()
example_read.head()

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

def list_data_files():
    return glob(os.path.join(ABSOLUTE_DATA_PATH, '*.csv'))
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
