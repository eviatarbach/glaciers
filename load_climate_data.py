import datetime
import pickle
from functools import lru_cache

import numpy
import netCDF4
from scipy.stats import linregress

from data import closest_index_in_range

YEAR_START, YEAR_END = 1960, 2014
NUM_YEARS = YEAR_END - YEAR_START + 1


@lru_cache(maxsize=512)
def lapse_rate(glacier_lat_index, glacier_lon_index):
    grid_square_temps = temp_var[:, :, glacier_lat_index, glacier_lon_index]
    grid_square_heights = height_var[:, :, glacier_lat_index, glacier_lon_index]

    lapse_rates = []

    for i in range(NUM_YEARS*12):
        heights = grid_square_heights[i, :]
        temps = grid_square_temps[i, :]

        index = heights < 10000  # restrict to troposphere

        lapse_rates.append(-linregress(heights[index], temps[index]).slope)

    lapse_mean = numpy.array(lapse_rates).reshape(NUM_YEARS, 12).mean(axis=1).mean()

    return lapse_mean

with open('data/serialized/glaciers_climate', 'br') as glaciers_file,\
     open('data/serialized/all_glaciers', 'br') as all_glaciers_file:
    glaciers = pickle.load(glaciers_file)
    all_glaciers = pickle.load(all_glaciers_file)

with netCDF4.Dataset('data/cru_ts3.23.1901.2014.tmp.dat.nc', 'r') as temp_data,\
     netCDF4.Dataset('data/cru_ts3.23.1901.2014.pre.dat.nc', 'r') as pre_data,\
     netCDF4.Dataset('data/cru_ts3.23.1901.2014.cld.dat.nc', 'r') as cld_data,\
     netCDF4.Dataset('data/air.mon.mean.nc', 'r') as lapse_temp,\
     netCDF4.Dataset('data/hgt.mon.mean.nc', 'r') as lapse_height:

    temp = temp_data.variables['tmp']
    pre = pre_data.variables['pre']
    cld = cld_data.variables['cld']

    days = numpy.array(list(temp_data.variables['time']))
    start = datetime.date(1900, 1, 1)  # dates are measured from here
    years = list(map(lambda d: (start + datetime.timedelta(days=d)).year, days))

    temp_var = lapse_temp.variables['air'][-(15 + NUM_YEARS*12):-15, :, :, :]
    height_var = lapse_height.variables['hgt'][-(15 + NUM_YEARS*12):-15, :, :, :]

    c = 0
    for glacier_set in [glaciers, all_glaciers]:
        for glacier in glacier_set.index:
            c += 1
            print(c/(len(glaciers) + len(all_glaciers)))
            glacier_lat_index = closest_index_in_range(-89.75, 89.75, 0.5,
                                                       glacier_set.loc[glacier, 'lat'])
            glacier_lon_index = closest_index_in_range(-179.75, 179.75, 0.5,
                                                       glacier_set.loc[glacier, 'lon'])

            grid_square_temps = temp[:, glacier_lat_index,
                                     glacier_lon_index].reshape([len(years)//12, 12])

            # Check for at least NUM_YEARS years of data
            if grid_square_temps.shape[0] >= NUM_YEARS:
                # TODO: check if correct
                continentality = (grid_square_temps[-NUM_YEARS:, :].max(axis=1).mean() -
                                  grid_square_temps[-NUM_YEARS:, :].min(axis=1).mean())

                glacier_set.loc[glacier, 'continentality'] = continentality

                if glacier_set.loc[glacier, 'lat'] > 0:
                    # Northern Hemisphere, use June, July, August
                    mean_summer_temp = grid_square_temps[-NUM_YEARS:, 5:8].mean()
                else:
                    # Southern Hemisphere, use December, January, February
                    mean_summer_temp = grid_square_temps[-NUM_YEARS:, [0, 1, 11]].mean()
                glacier_set.loc[glacier, 'summer_temperature'] = mean_summer_temp

            grid_square_pre = pre[:, glacier_lat_index, glacier_lon_index].reshape([len(years)//12,
                                                                                    12])

            # Check for at least NUM_YEARS years of data
            if grid_square_pre.shape[0] >= NUM_YEARS:
                mean_pre = grid_square_pre[-NUM_YEARS:, :].sum(axis=1).mean()
                glacier_set.loc[glacier, 'precipitation'] = mean_pre

                if glacier_set.loc[glacier, 'lat'] > 0:
                    # Northern Hemisphere, use December, January, February
                    mean_winter_pre = grid_square_pre[-NUM_YEARS:, [0, 1, 11]].sum(axis=1).mean()
                else:
                    # Southern Hemisphere, use June, July, August
                    mean_winter_pre = grid_square_pre[-NUM_YEARS:, 5:8].sum(axis=1).mean()
                glacier_set.loc[glacier, 'winter_precipitation'] = mean_winter_pre

            grid_square_cld = cld[:, glacier_lat_index, glacier_lon_index].reshape([len(years)//12,
                                                                                    12])

            # Check for at least NUM_YEARS years of data
            if grid_square_cld.shape[0] >= NUM_YEARS:
                mean_cld = grid_square_cld[-NUM_YEARS:, :].mean()
                glacier_set.loc[glacier, 'cloud_cover'] = mean_cld

            glacier_lat_index = 72 - closest_index_in_range(-90, 90, 2.5,
                                                            glacier_set.loc[glacier, 'lat'])
            lon = glacier_set.loc[glacier, 'lon']

            # conversion between 0 to 360 and -180 to 180 degree system
            glacier_lon_index = (closest_index_in_range(-180, 180, 2.5,
                                                        lon if lon < 180
                                                        else -(180 - (lon - 180))) + 72) % 144

            glacier_set.loc[glacier, 'lapse_rate'] = lapse_rate(glacier_lat_index,
                                                                glacier_lon_index)

all_glaciers.to_pickle('data/serialized/all_glaciers')
glaciers.to_pickle('data/serialized/glaciers_climate')
