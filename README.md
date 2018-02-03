# Code for "How sensitive are mountain glaciers to climate change? Insights from a block model"

This repository contains the code for the paper "How sensitive are mountain glaciers to climate change? Insights from a block model" by Eviatar Bach, Valentina Radić, and Christian Schoof, 2018, to be published in the *Journal of Glaciology*.

The code is for computing response times and sensitivity to equilibrium line altitude (ELA) changes for mountain glaciers worldwide, using a block model with volume–area–length scaling and a piecewise linear mass balance profile.

All the code was written by Eviatar Bach. Licensed under the GNU Public License v3.0.

You can contact me with any questions at eviatarbach@protonmail.com.

## Setup instructions
The easiest way to install the dependencies is with [conda](https://conda.io/docs/). After installing that and cloning this repository, the following instructions should get you a working local installation of the project.

1. Install the dependencies using the conda environment: `conda env create -f environment.yml`
2. Switch into the project environment: `source activate glaciers`
3. You will need to install the R sensitivity package, since it is not installed with the conda environment. In either the command-line interface `R` or in RStudio, run `install.packages("sensitivity")`. This will download and compile the package.

That's it! Now you can run any of the files in the repository (see *Description of files* below). The serialized data files are included, so you do not have to run the `load_*.py` files. If you want to, see below.

## Data sources
Except for the glacier thickness estimates, the other data must be downloaded and extracted as described to allow the data loading to work.

* Glacier thickness estimates kindly provided by Matthias Huss (in `data/thick`), based on [Huss & Farinotti 2012](http://doi.org/10.1029/2012JF002523)
* [Randolph Glacier Inventory 5.0](https://www.glims.org/RGI/rgi50_files/rgi50.zip) (extract the top directory and the sub-archives into `data`)
* [Fluctuations of Glaciers Database](http://wgms.ch/downloads/DOI-WGMS-FoG-2017-06.zip) (extract into `data/DOI-WGMS-FoG-2017-06`)
* [CRU TS v. 3.22](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/cruts.1506241137.v3.23/) (extract [`cld`](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/cruts.1506241137.v3.23/cld/cru_ts3.23.1901.2014.cld.dat.nc.gz), [`pre`](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/cruts.1506241137.v3.23/pre/cru_ts3.23.1901.2014.pre.dat.nc.gz), and [`tmp`](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/cruts.1506241137.v3.23/tmp/cru_ts3.23.1901.2014.tmp.dat.nc.gz) into `data`)
* [NCEP/NCAR Reanalysis](https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.derived.pressure.html) (download [monthly air temperature](ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/air.mon.mean.nc) and [monthly geopotential height](ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/hgt.mon.mean.nc) into `data`)


## Libraries used
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

## Description of files
- **data.py**: contains various constants, including the scaling constant, uncertainty values for different parameters, and functions for computing equilibrium volumes
- **environment.yml**: conda environment for the project
- **hsic.py**: contains the functions for calling the R library `sensitivity` to compute the Hilbert--Schmidt independence criterion
- **interpolate_missing.py**: interpolates volumes and lengths for glaciers that are missing them. Notably, these are interpolated for all the glaciers in Alaska and Southern Andes, due to a mismatch in numbering between the Randolph Glacier Inventory 5.0 and the Huss & Farinotti data 
- **load_climate.py**: load climate data to be used for estimating mass-balance gradients
- **load_data.py**: calls all the other data-loading scripts in the correct order. Running this will load and serialize all the data necessary for running the model.
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
