# Reports Directory

This directory contains reports, presentations, and generated figures from our analysis.

## Figures

The `figures/` directory is used to store generated plots and visualizations. This directory is ignored by git since the figures can be regenerated from the source code.

To regenerate all figures:

1. Make sure you have the processed sensor data in `data/interim/sensor_data_resampled.pkl`
2. Run the visualization script:
   ```bash
   python visualization/visualize.py
   ```

This will create all sensor data plots in the `figures/` directory.
