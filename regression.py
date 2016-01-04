import csv
import datetime
import bisect

import pandas
import numpy
import scipy.io.netcdf as netcdf

# Used for missing values in NetCDF
NETCDF_FILL_VALUE = 9.969209968386869e+36

data = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-EE-MASS-BALANCE.csv',
                       usecols=['WGMS_ID', 'YEAR', 'LOWER_BOUND', 'UPPER_BOUND',
                                'ANNUAL_BALANCE'],
                       index_col=['WGMS_ID', 'YEAR'],
                       encoding='ISO-8859-1')

data['ALTITUDE'] = (data['LOWER_BOUND'] + data['UPPER_BOUND'])/2

data = data.drop(['LOWER_BOUND', 'UPPER_BOUND'], axis=1)

# Only use mass balance data by altitude band
data = data[data['ALTITUDE'] != 9999]
data = data.dropna()

names = data.index.get_level_values('WGMS_ID').unique()

glaciers = pandas.DataFrame({'WGMS ID': names})
glaciers = glaciers.set_index('WGMS ID')

data_latlon = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-A-GENERAL-INFORMATION.csv',
                              usecols=['WGMS_ID', 'LATITUDE', 'LONGITUDE'],
                              index_col='WGMS_ID', encoding='ISO-8859-1')

glaciers[['lat', 'lon']] = data_latlon.loc[names]

data = data.sort_index(level=[0, 1])

# Calculate average mass balance gradients
for glacier in names:
    gradients = []
    for year in data.loc[glacier].index.unique():
        array = data.loc[glacier, year].values
        altitudes = numpy.hstack([numpy.ones([array.shape[0], 1]), array[:, 1:]])
        balance = array[:, 0:1]

        # Linear regression, g is the slope
        g = numpy.linalg.lstsq(altitudes, balance)[0][1]
        gradients.append(g)
    glaciers.loc[glacier, 'g'] = numpy.mean(gradients)

temp_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.tmp.dat.nc', 'r')
pre_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.pre.dat.nc', 'r')

lat = numpy.array(list(temp_data.variables['lat']))
lon = numpy.array(list(temp_data.variables['lon']))
temp = temp_data.variables['tmp']

pre = pre_data.variables['pre']

days = numpy.array(list(temp_data.variables['time']))
start = datetime.date(1900, 1, 1)  # dates are measured from here
years = list(map(lambda d: (start + datetime.timedelta(days=d)).year, days))

for glacier in names:
    # TODO: this can easily be done in constant time
    glacier_lat_index = bisect.bisect(lat, glaciers.loc[glacier, 'lat'])
    glacier_lon_index = bisect.bisect(lon, glaciers.loc[glacier, 'lon'])

    grid_square_temps = temp[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_temps = grid_square_temps[grid_square_temps[:, 0] != NETCDF_FILL_VALUE, :]

    # Check for at least 10 years of data
    if grid_square_temps.shape[0] >= 20:
        continentality = grid_square_temps[-20:, :].max(axis=1).mean() - grid_square_temps[-20:, :].min(axis=1).mean()
        glaciers.loc[glacier, 'continentality'] = continentality

        if glaciers.loc[glacier, 'lat'] > 0:  # Northern Hemisphere, use June, July, August
            mean_summer_temp = grid_square_temps[-20:, 5:8].mean()
        else:  # Southern Hemisphere, use December, January, February
            mean_summer_temp = grid_square_temps[-20:, [0, 1, 11]].mean()
        glaciers.loc[glacier, 'summer temperature'] = mean_summer_temp

    grid_square_pre = pre[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_pre = grid_square_pre[grid_square_pre[:, 0] != NETCDF_FILL_VALUE, :]

    # Check for at least 10 years of data
    if grid_square_pre.shape[0] >= 20:
        mean_pre = grid_square_pre[-20:, :].sum(axis=1).mean()
        glaciers.loc[glacier, 'precipitation'] = mean_pre

        if glaciers.loc[glacier, 'lat'] > 0:  # Northern Hemisphere, use December, January, February
            mean_winter_pre = grid_square_pre[-20:, [0, 1, 11]].mean()
        else:  # Southern Hemisphere, use June, July, August
            mean_winter_pre = grid_square_pre[-20:, 5:8].mean()
        glaciers.loc[glacier, 'winter precipitation'] = mean_winter_pre
