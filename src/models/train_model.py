import numpy as np
import pandas as pd
import pickle
import os

# Change to models directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

subset_features = df_features.drop(columns=["participant", "exercise_category", "set"])

X = subset_features.drop(columns=["exercise_name"])
y = subset_features["exercise_name"]

# stratify ensure equal distribution of exercises in the train-test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Visualize the distribution of exercises in the train-test set
fig, ax = plt.subplots(figsize=(10, 5))
subset_features["exercise_name"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(
    kind="bar", ax=ax, color="dodgerblue", label="Train"
)
y_test.value_counts().plot(
    kind="bar", ax=ax, color="royalblue", label="Test"
)
plt.legend()
plt.show()


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
