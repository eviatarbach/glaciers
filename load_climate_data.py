import pickle
import datetime

import numpy
import scipy.io.netcdf as netcdf

from data import closest_index_in_range

all_glaciers = pickle.load(open('data/serialized/all_glaciers.p', 'br'))

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

for glacier in all_glaciers.index:
    glacier_lat_index = closest_index_in_range(-89.75, 89.75, 0.5,
                                               all_glaciers.loc[glacier,
                                                                'Location_y'])
    glacier_lon_index = closest_index_in_range(-179.75, 179.75, 0.5,
                                               all_glaciers.loc[glacier,
                                                                'Location_x'])

    grid_square_temps = temp[:, glacier_lat_index,
                             glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_temps = grid_square_temps[grid_square_temps[:, 0] !=
                                          fill_value,
                                          :]

    # Check for at least 20 years of data
    if grid_square_temps.shape[0] >= 20:
        # TODO: check if correct
        continentality = (grid_square_temps[-20:, :].max(axis=1).mean() -
                          grid_square_temps[-20:, :].min(axis=1).mean())
        all_glaciers.loc[glacier, 'continentality'] = continentality

        if all_glaciers.loc[glacier, 'Location_y'] > 0:
            # Northern Hemisphere, use June, July, August
            mean_summer_temp = grid_square_temps[-20:, 5:8].mean()
        else:
            # Southern Hemisphere, use December, January, February
            mean_summer_temp = grid_square_temps[-20:, [0, 1, 11]].mean()
        all_glaciers.loc[glacier, 'summer_temperature'] = mean_summer_temp

    grid_square_pre = pre[:, glacier_lat_index,
                          glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_pre = grid_square_pre[grid_square_pre[:, 0] != fill_value, :]

    # Check for at least 20 years of data
    if grid_square_pre.shape[0] >= 20:
        mean_pre = grid_square_pre[-20:, :].sum(axis=1).mean()
        all_glaciers.loc[glacier, 'precipitation'] = mean_pre

        if all_glaciers.loc[glacier, 'Location_y'] > 0:
            # Northern Hemisphere, use December, January, February
            mean_winter_pre = grid_square_pre[-20:,
                                              [0, 1, 11]].sum(axis=1).mean()
        else:
            # Southern Hemisphere, use June, July, August
            mean_winter_pre = grid_square_pre[-20:, 5:8].sum(axis=1).mean()
        all_glaciers.loc[glacier, 'winter_precipitation'] = mean_winter_pre

    grid_square_cld = cld[:, glacier_lat_index,
                          glacier_lon_index].reshape([len(years)/12, 12])

    # Get rid of all years with missing data
    grid_square_cld = grid_square_cld[grid_square_cld[:, 0] != fill_value, :]

    # Check for at least 20 years of data
    if grid_square_cld.shape[0] >= 20:
        mean_cld = grid_square_cld[-20:, :].mean()
        all_glaciers.loc[glacier, 'cloud_cover'] = mean_cld

pickle.dump(all_glaciers, open('all_glaciers.p', 'wb'))
