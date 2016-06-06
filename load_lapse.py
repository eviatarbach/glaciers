import pickle

import numpy
import scipy
import scipy.stats
import netCDF4

from data import closest_index_in_range


def straddle(arr, value):
    diffs = arr - value
    diffs[diffs < 0] = float('inf')
    index = numpy.argmin(diffs, axis=1)

    # Boundary cases
    index[value < arr[:, 0]] = 1
    index[value > arr[:, -1]] = arr.shape[1] - 1
    return numpy.vstack([index - 1, index]).T

temp = netCDF4.Dataset('air.mon.mean.nc', 'a')  # monthly means of air temperatures
height = netCDF4.Dataset('hgt.mon.mean.nc', 'a')  # geopotential heights

# slice 1995--2014
temp_var = temp.variables['air'][-255:-15, :, :, :]
height_var = height.variables['hgt'][-255:-15, :, :, :]

all_glaciers = pickle.load(open('all_glaciers.p', 'br'))

unique = set()

for i,  glacier in enumerate(all_glaciers.index):
    if i % 500 == 0:
        print(i/len(all_glaciers))

    glacier_lat_index = 72 - closest_index_in_range(-90, 90, 2.5, all_glaciers.loc[glacier, 'Location_y'])
    lon = all_glaciers.loc[glacier, 'Location_x']

    # conversion between 0 to 360 and -180 to 180 degree system
    glacier_lon_index = (closest_index_in_range(-180, 180, 2.5, lon if lon < 180 else -(180 - (lon - 180))) + 72) % 144

    unique.add((glacier_lat_index, glacier_lon_index))

    grid_square_temps = temp_var[:, :, glacier_lat_index, glacier_lon_index]
    grid_square_heights = height_var[:, :, glacier_lat_index, glacier_lon_index]

    elev = all_glaciers.loc[glacier, 'Zmed']

    #straddling = straddle(grid_square_heights, elev)
    #elevs = grid_square_heights[:, straddling][:, 0]
    #temps = grid_square_temps[:, straddling][:, 0]
    #lapse_rates = -(temps[:, 1] - temps[:, 0])/(elevs[:, 1] - elevs[:, 0])

    lapse_rates = []

    for i in range(240):
        heights = grid_square_heights[i, :]
        temps = grid_square_temps[i, :]

        index = heights < 10000
        lapse_rates.append(-scipy.stats.linregress(heights[index], temps[index])[0])

    lapse_mean = numpy.array(lapse_rates).reshape(20, 12).mean(axis=1).mean()

    '''if (lapse_mean < 0) and (glacier[0] not in ['Alaska']):
        from IPython.core.debugger import Tracer
        Tracer()()'''

    all_glaciers.loc[glacier, 'lapse_rate'] = lapse_mean
