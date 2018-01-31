# Code for "How sensitive are mountain glaciers to climate change? Insights from a block model"

This repository contains the code for the paper "How sensitive are mountain glaciers to climate change? Insights from a block model" by Eviatar Bach, Valentina Radić, and Christian Schoof, 2018, to be published in the *Journal of Glaciology*.

The code is for computing response times and sensitivity to equilibrium line altitude (ELA) changes for mountain glaciers worldwide, using a block model with volume–area–length scaling and a piecewise linear mass balance profile.

All the code was written by Eviatar Bach. Licensed under the GNU Public License v3.0.

Libraries used:
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [geopandas](http://geopandas.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/scipylib/index.html)
- [uncertainties](https://pythonhosted.org/uncertainties/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [netcdf4-python](http://unidata.github.io/netcdf4-python/)
- [sensitivity](https://cran.r-project.org/web/packages/sensitivity/index.html)
- [mpl-scatter-density](https://github.com/astrofrog/mpl-scatter-density)
- [astropy](http://www.astropy.org/)
- [rpy2](https://rpy2.bitbucket.io/)

Description of files:
- **data.py**: contains various constants, including the scaling constant, uncertainty values for different parameters, and functions for computing equilibrium volumes
- **hsic.py**: contains the functions for calling the R library `sensitivity` to compute the Hilbert--Schmidt independence criterion
- **interpolate_missing.py**: interpolates volumes and lengths for glaciers that are missing them. Notably, these are interpolated for all the glaciers in Alaska and Southern Andes, due to a mismatch in numbering between the Randolph Glacier Inventory 5.0 and the Huss & Farinotti data 
- **load_climate.py**: load climate data to be used for estimating mass-balance gradients
- **load_data.py**: calls all the other data-loading scripts. Running this will load and serialize all the data necessary for running the model.
- **load_geometry.py**: load geometry data from the RGI and Matthias Huss's thickness estimates
- **load_gradients.py**: estimates mass-balance gradients for all glaciers in the data set. The variables used in the regression are specified at the top of the file, and were selected using `subset_selection.py`.
- **load_mass_balance.py**: estimate mass-balance gradients for glaciers that have mass-balance data provided by WGMS
- **plot_HSIC.py**: generates Figure 6 in the paper
- **plot_bifurcation_2D.py**: generates Figures 3 (center) and 3 (right) in the paper
- **plot_bifurcation_3D.py**: generates Figure 3 (left) in the paper
- **plot_config.py**: sets plot font configuration. If you do not have the Optima font, comment out line 15.
- **plot_curves.py**: generates Figure 2 in the paper
- **plot_density.py**: generates Figure 5 in the paper
- **plot_sensitivity.py**: generates Figure 4 (left) in the paper
- **plot_slope_uncertainty.py**: generates Figure 7 in the paper
- **plot_tau.py**: generates Figure 4 (right) in the paper
- **regional_diff.py**: calculates the correlations between regional mean quantities and the regional sensitivities and response times
- **roots.py**: functions for finding equilibrium values of the model
- **sample.py**: sampling functions for use with the HSIC analysis
- **sensitivity_climate.py**: compute the sensitivities and response times for all glaciers in the data set
- **subset_selection.py**: finds the set of variables to use for predicting mass-balance gradients
- **verify_derivation.nb**: a Mathematica notebook for verifying the derivation of the nondimensionalized equation
