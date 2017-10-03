# Code for "How sensitive are mountain glaciers to climate change? Insights from a block model"

This repository contains the code for the paper "How sensitive are mountain glaciers to climate change? Insights from a block model" by Eviatar Bach, Valentina Radić, and Christian Schoof, 2017, to be published in the *Journal of Glaciology*.

The code is for computing response times and sensitivity to equilibrium line altitude (ELA) changes for mountain glaciers worldwide, using a block model with volume–area–length scaling and a piecewise linear mass balance profile.

All the code was written by Eviatar Bach. Licensed under the GNU Public License v3.0.

Libraries used:
- NumPy
- pandas
- geopandas
- matplotlib
- scipy
- uncertainties
- scikit-learn
- netcdf4-python
- sensitivity

Description of files:
- **data.py**: contains various constants, including the scaling constant, uncertainty values for different parameters, and functions for computing equilibrium volumes
- **interpolate_missing.py**: interpolates volumes and lengths for glaciers that are missing them. Notably, these are interpolated for all the glaciers in Alaska and Southern Andes, due to a mismatch in numbering between the Randolph Glacier Inventory 5.0 and the Huss & Farinotti data 
- **load_climate.py**:
