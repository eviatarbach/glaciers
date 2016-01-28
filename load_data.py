import csv
import datetime
import bisect

import pandas
import numpy
import scipy.io.netcdf as netcdf

def closest_index_in_range(lower, upper, step, value):
    '''
    Find the index of the closest value to `value` in the range
    [lower, lower + step, ..., upper - step, upper] in constant time. `upper`
    must be greater than `lower`. If `value` is outside the range, return the
    corresponding boundary index (0 or the last index). When two values are
    equally close, the index of the smaller is returned.
    '''
    if value >= upper:
        return int((upper - lower)/step)
    elif value < lower:
        return 0

    value = value - lower
    upper = upper - lower
    lower = 0

    index = int(value//step + 1)

    if step*index - value >= value - step*(index - 1):
        index -= 1

    return index

data_mb = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-EE-MASS-BALANCE.csv',
                          usecols=['WGMS_ID', 'YEAR', 'LOWER_BOUND', 'UPPER_BOUND',
                                   'ANNUAL_BALANCE'],
                          index_col=['WGMS_ID', 'YEAR'],
                          encoding='ISO-8859-1')

data_mb['ALTITUDE'] = (data_mb['LOWER_BOUND'] + data_mb['UPPER_BOUND'])/2

data_mb = data_mb.drop(['LOWER_BOUND', 'UPPER_BOUND'], axis=1)

# Only use mass balance data by altitude band
data_mb = data_mb[data_mb['ALTITUDE'] != 9999]
data_mb = data_mb.dropna()

names = data_mb.index.get_level_values('WGMS_ID').unique()

glaciers = pandas.DataFrame({'WGMS_ID': names})
glaciers = glaciers.set_index('WGMS_ID')

data_latlon = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-A-GENERAL-INFORMATION.csv',
                              usecols=['WGMS_ID', 'LATITUDE', 'LONGITUDE'],
                              index_col='WGMS_ID', encoding='ISO-8859-1')

data_elevation = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-B-STATE.csv',
                                 usecols=['WGMS_ID', 'YEAR', 'HIGHEST_ELEVATION', 'MEDIAN_ELEVATION'],
                                 index_col=['WGMS_ID', 'YEAR'], encoding='ISO-8859-1')
data_elevation = data_elevation.dropna().groupby(level=0).mean()

glaciers[['lat', 'lon']] = data_latlon.loc[names]

glaciers[['max_elevation', 'median_elevation']] = data_elevation.loc[names]

data_mb = data_mb.sort_index(level=[0, 1])

data_ela = pandas.read_csv('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-E-MASS-BALANCE-OVERVIEW.csv',
                           usecols=['WGMS_ID', 'YEAR', 'ELA_PREFIX', 'ELA'],
                           index_col=['WGMS_ID', 'YEAR'], encoding='ISO-8859-1')

data_ela = data_ela.sort_index(level=[0, 1])

data_ela = data_ela.dropna(subset=['ELA'])

# Want only rows where ELA_PREFIX is NaN; otherwise, this means that the ELA is
# above or below the entire glacier
data_ela = data_ela[data_ela['ELA_PREFIX'].isnull()]

data_mb['ela'] = data_ela['ELA']
data_mb = data_mb.dropna()

# Calculate average mass balance gradients
for glacier in names:
    gradients = []
    gradients2 = []
    gradients3 = []
    for year in data_mb.loc[glacier].index.unique():
        # Gradient using slope of linear regression
        array = data_mb.loc[glacier, year][['ANNUAL_BALANCE', 'ALTITUDE']].values
        altitudes = numpy.hstack([numpy.ones([array.shape[0], 1]), array[:, 1:]])
        balance = array[:, 0:1]

        g = numpy.linalg.lstsq(altitudes, balance)[0][1]
        gradients.append(g)

        # Gradient around ELA
        ela = data_mb.loc[glacier, year]['ela'].iloc[0]

        if array[0, 1] < ela < array[-1, 1]:
            ela_i = bisect.bisect(array[:, 1:], ela)

            y2 = array[ela_i - 1:ela_i + 1, 1][1]
            y1 = array[ela_i - 1:ela_i + 1, 1][0]

            mb2 = array[ela_i - 1:ela_i + 1, 0][1]
            mb1 = array[ela_i - 1:ela_i + 1, 0][0]

            g2 = (mb2 - mb1)/(y2 - y1)
            gradients2.append(g2)

        # Average gradient from lowest and highest altitude
        y2 = array[-1, 1]
        y1 = array[0, 1]

        mb2 = array[-1, 0]
        mb1 = array[0, 0]

        g3 = (mb2 - mb1)/(y2 - y1)
        gradients3.append(g3)

    glaciers.loc[glacier, 'g'] = numpy.mean(gradients)
    glaciers.loc[glacier, 'g2'] = numpy.mean(gradients2)
    glaciers.loc[glacier, 'g3'] = numpy.mean(gradients3)

temp_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.tmp.dat.nc', 'r')
pre_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.pre.dat.nc', 'r')
cld_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.cld.dat.nc', 'r')

lat = numpy.array(list(temp_data.variables['lat']))
lon = numpy.array(list(temp_data.variables['lon']))

temp = temp_data.variables['tmp']
pre = pre_data.variables['pre']
cld = cld_data.variables['cld']

fill_value = temp.missing_value
assert(fill_value == pre.missing_value == cld.missing_value)

days = numpy.array(list(temp_data.variables['time']))
start = datetime.date(1900, 1, 1)  # dates are measured from here
years = list(map(lambda d: (start + datetime.timedelta(days=d)).year, days))

for glacier in names:
    glacier_lat_index = closest_index_in_range(-89.75, 89.75, 0.5, glaciers.loc[glacier, 'lat'])
    glacier_lon_index = closest_index_in_range(-179.75, 179.75, 0.5, glaciers.loc[glacier, 'lon'])

    grid_square_temps = temp[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_temps = grid_square_temps[grid_square_temps[:, 0] != fill_value, :]

    # Check for at least 20 years of data
    if grid_square_temps.shape[0] >= 20:
        continentality = grid_square_temps[-20:, :].max(axis=1).mean() - grid_square_temps[-20:, :].min(axis=1).mean()
        glaciers.loc[glacier, 'continentality'] = continentality

        if glaciers.loc[glacier, 'lat'] > 0:  # Northern Hemisphere, use June, July, August
            mean_summer_temp = grid_square_temps[-20:, 5:8].mean()
        else:  # Southern Hemisphere, use December, January, February
            mean_summer_temp = grid_square_temps[-20:, [0, 1, 11]].mean()
        glaciers.loc[glacier, 'summer_temperature'] = mean_summer_temp

    grid_square_pre = pre[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_pre = grid_square_pre[grid_square_pre[:, 0] != fill_value, :]

    # Check for at least 20 years of data
    if grid_square_pre.shape[0] >= 20:
        mean_pre = grid_square_pre[-20:, :].sum(axis=1).mean()
        glaciers.loc[glacier, 'precipitation'] = mean_pre

        if glaciers.loc[glacier, 'lat'] > 0:  # Northern Hemisphere, use December, January, February
            mean_winter_pre = grid_square_pre[-20:, [0, 1, 11]].mean()
        else:  # Southern Hemisphere, use June, July, August
            mean_winter_pre = grid_square_pre[-20:, 5:8].mean()
        glaciers.loc[glacier, 'winter_precipitation'] = mean_winter_pre

    grid_square_cld = cld[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_cld = grid_square_cld[grid_square_cld[:, 0] != fill_value, :]

    # Check for at least 20 years of data
    if grid_square_cld.shape[0] >= 20:
        mean_cld = grid_square_cld[-20:, :].mean()
        glaciers.loc[glacier, 'cloud_cover'] = mean_cld

glaciers = glaciers.dropna()
glaciers.to_pickle('glaciers')
