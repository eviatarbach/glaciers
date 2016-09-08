from data import closest_index_in_range

import datetime

import numpy
import pandas
import netCDF4
import matplotlib.pyplot as plt
from scipy.stats import linregress

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
glaciers_ela = pandas.read_pickle('data/serialized/glaciers_ela')

glaciers_ela = glaciers_ela.sortlevel()

# Filter out glaciers that less than fifteen year of data
glaciers_ela = glaciers_ela[glaciers_ela.count(level='WGMS_ID') >= 15].dropna()

lapses = []
empirical_lapses = []

dT = []
dELA = []

with netCDF4.Dataset('data/cru_ts3.23.1901.2014.tmp.dat.nc', 'r') as temp_data:
    temp = temp_data.variables['tmp']
    days = numpy.array(list(temp_data.variables['time']))
    start = datetime.date(1900, 1, 1)  # dates are measured from here
    years = list(map(lambda d: (start + datetime.timedelta(days=d)).year, days))

    for glacier in glaciers_ela.reset_index()['WGMS_ID'].unique():
        values = glaciers_ela.loc[glacier].reset_index().values
        glacier_lat_index = closest_index_in_range(-89.75, 89.75, 0.5,
                                                   glaciers.loc[glacier, 'lat'])
        glacier_lon_index = closest_index_in_range(-179.75, 179.75, 0.5,
                                                   glaciers.loc[glacier, 'lon'])

        grid_square_temps = temp[:, glacier_lat_index,
                                 glacier_lon_index].reshape([len(years)//12, 12])

        temps = grid_square_temps.mean(axis=1)[numpy.searchsorted(numpy.array(years)[range(0, len(years), 12)], values[:, 0])]
        altitudes = values[:, 1]
        regression = linregress(temps, altitudes)
        dELA.append(regression.slope*temps[-1] + regression.intercept - (regression.slope*temps[0] + regression.intercept))
        dT.append(temps[-1] - temps[0])
        if regression.pvalue <= 0.05:
            lapses.append(glaciers.loc[glacier, 'lapse_rate'])
            empirical_lapses.append(1/regression.slope)
            glaciers.loc[glacier, 'diff'] = glaciers.loc[glacier, 'lapse_rate'] - regression.slope
            # print(regression.slope)
            # plt.scatter(temps, altitudes)
            # plt.plot(temps, regression.slope*temps + regression.intercept)
            # plt.show()
