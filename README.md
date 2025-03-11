# Fitness Tracker App

# Barbell Exercise Classification and Rep Counter

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Project Overview

This project aims to create Python scripts to process, visualize, and model accelerometer and gyroscope data to create a machine learning model that can classify barbell exercises and count repetitions. The system will analyze sensor data to automatically identify different barbell movements and track exercise performance.

<div align="center">
  <img src="images/barbell_exercises.png" alt="Common barbell exercises: Bench Press, Deadlift, Overhead Press, Barbell Row, and Squat" width="800"/>
</div>

### Quantified Self

This project aligns with the principles of the Quantified Self movement: "The quantified self is any individual engaged in the self-tracking of any kind of biological, physical, behavioral, or environmental information. The self-tracking is driven by a certain goal of the individual, with a desire to act upon the collected information" (Hoogendoorn, M., & Funk, B., 2018). By tracking and analyzing exercise data, users can gain insights into their workout patterns, improve form, and track progress over time.

### Data Source

The raw sensor data used in this project can be downloaded from the [Datalumina documentation portal](https://docs.datalumina.io/s1eFpcdtbvRTMb). The dataset includes accelerometer and gyroscope readings from various barbell exercises, including bench press, deadlift, overhead press, barbell row, and squat. After downloading, place the data files in the `data/raw` directory.

Note: The data directories contain `.gitkeep` files to maintain the directory structure in version control, even though the actual data files are excluded via `.gitignore`. This ensures that the project structure remains intact when cloning the repository.

## Environment Setup

This project supports two methods for managing dependencies:

### Option 1: Python venv (Recommended)

This is a lightweight option using Python's built-in virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment

If you prefer using Conda, you can use the provided `environment.yml`:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate tracking-barbell-exercises
```

Choose the option that best suits your workflow. The venv approach is recommended for its simplicity and built-in Python support.

## Interactive Python Development

This project supports interactive Python development using Jupyter functionality in VS Code. To set this up:

1. Install the Jupyter kernel for your virtual environment:

```bash
# With your virtual environment activated
python -m ipykernel install --user --name=tracking-barbell-exercises
```

2. Configure VS Code:

   - Press `Cmd+Shift+P` (on macOS) or `Ctrl+Shift+P` (on Windows)
   - Type "Python: Select Interpreter"
   - Choose the `tracking-barbell-exercises` environment

3. Enable interactive Python mode:

   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows)
   - Choose one of:
     - "Jupyter: Create New Blank Notebook"
     - "Jupyter: Create Interactive Window"
     - Click the "Jupyter: Run Current File in Interactive Window" button (top-right corner)

4. Use Shift+Enter to run code cells interactively

Alternatively, you can use traditional Jupyter notebooks located in the `notebooks` directory for data exploration and analysis.

## Data Management

This project follows best practices for managing large sensor data files. The `/data/` directory is excluded from version control via `.gitignore`:

```plaintext
# exclude data from source control by default
/data/
```

Typically, you want to exclude this folder if it contains either sensitive data that you do not want to add to version control or large files.

## Duplicating the .env File

To set up your environment variables, you need to duplicate the `.env.example` file and rename it to `.env`. You can do this manually or using the following terminal command:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py
```

---
