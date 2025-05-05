import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# Define project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")

# Load the feature data
def load_feature_data():
    feature_file = os.path.join(DATA_INTERIM, "03_data_features.pkl")
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the data
df_features = load_feature_data()
print(f"Loaded feature data with shape: {df_features.shape}")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
