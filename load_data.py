import datetime

import pandas
import numpy
from scipy.stats import linregress
import netCDF4

from data import closest_index_in_range

YEAR_START, YEAR_END = 1960, 2014
NUM_YEARS = YEAR_END - YEAR_START + 1

with open('data/DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-EE-MASS-BALANCE.csv', 'r',
          encoding='ISO-8859-1') as mb_file,\
     open('data/DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-A-GENERAL-INFORMATION.csv', 'r',
          encoding='ISO-8859-1') as latlon_file,\
     open('data/DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-B-STATE.csv', 'r',
          encoding='ISO-8859-1') as elev_file,\
     open('data/DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-E-MASS-BALANCE-OVERVIEW.csv', 'r',
          encoding='ISO-8859-1') as ela_file:

    data_mb = pandas.read_csv(mb_file, index_col=['WGMS_ID', 'YEAR'],
                              usecols=['WGMS_ID', 'YEAR', 'LOWER_BOUND', 'UPPER_BOUND',
                                       'ANNUAL_BALANCE'])

    data_mb['ALTITUDE'] = (data_mb['LOWER_BOUND'] + data_mb['UPPER_BOUND'])/2

    data_mb = data_mb.drop(['LOWER_BOUND', 'UPPER_BOUND'], axis=1)

    # Only use mass balance data by altitude band
    data_mb = data_mb[data_mb['ALTITUDE'] != 9999]
    # Removing invalid value
    data_mb = data_mb[data_mb['ANNUAL_BALANCE'] != -9999]
    data_mb = data_mb.dropna()

    names = data_mb.index.get_level_values('WGMS_ID').unique()

    glaciers = pandas.DataFrame({'WGMS_ID': names})
    glaciers = glaciers.set_index('WGMS_ID')

    data_latlon = pandas.read_csv(latlon_file, index_col='WGMS_ID',
                                  usecols=['WGMS_ID', 'LATITUDE', 'LONGITUDE'])

    data_elevation = pandas.read_csv(elev_file, index_col=['WGMS_ID', 'YEAR'],
                                     usecols=['WGMS_ID', 'YEAR', 'HIGHEST_ELEVATION',
                                              'MEDIAN_ELEVATION'])
    data_elevation = data_elevation.dropna().groupby(level=0).mean()

    glaciers[['lat', 'lon']] = data_latlon.loc[names]

    glaciers[['max_elevation', 'median_elevation']] = data_elevation.loc[names]

    data_mb = data_mb.sort_index(level=[0, 1])

    data_ela = pandas.read_csv(ela_file, index_col=['WGMS_ID', 'YEAR'],
                               usecols=['WGMS_ID', 'YEAR', 'ELA_PREFIX', 'ELA'])

    data_ela = data_ela.sort_index(level=[0, 1])

    data_ela = data_ela.dropna(subset=['ELA'])

    # Want only rows where ELA_PREFIX is NaN; otherwise, this means that
    # the ELA is above or below the entire glacier
    data_ela = data_ela[data_ela['ELA_PREFIX'].isnull()]

    data_mb['ela'] = data_ela['ELA']
    data_mb = data_mb.dropna()

    # Calculate average mass balance gradients
    for glacier in names:
        gradients = []
        gradients_abl = []
        gradients_acc = []
        years = data_mb.loc[glacier].index.unique()
        years = years[(YEAR_START <= years) & (years <= YEAR_END)]
        for year in years:
            # Gradient using slope of linear regression
            array = data_mb.loc[glacier, year][['ANNUAL_BALANCE', 'ALTITUDE']].values
            altitudes = numpy.hstack([numpy.ones([array.shape[0], 1]),
                                      array[:, 1:]])
            balance = array[:, 0:1]

            g = numpy.linalg.lstsq(altitudes, balance)[0][1]
            gradients.append(g)

            ela = data_mb.loc[glacier, year]['ela'].iloc[0]
            ela_i = altitudes[:, 1].searchsorted(ela)

            # At least 4 mass balance measurements on either side of
            # the ELA
            if (ela_i >= 4) and (len(balance) - ela_i >= 4):
                g_abl = numpy.linalg.lstsq(altitudes[:ela_i], balance[:ela_i])[0][1]
                g_acc = numpy.linalg.lstsq(altitudes[ela_i:], balance[ela_i:])[0][1]

                gradients_abl.append(g_abl)
                gradients_acc.append(g_acc)

        glaciers.loc[glacier, 'g'] = numpy.mean(gradients)
        glaciers.loc[glacier, 'g_abl'] = numpy.mean(gradients_abl)
        glaciers.loc[glacier, 'g_acc'] = numpy.mean(gradients_acc)
        glaciers.loc[glacier, 'g_abl_std'] = numpy.std(gradients_abl)
        glaciers.loc[glacier, 'g_acc_std'] = numpy.std(gradients_acc)

with netCDF4.Dataset('data/cru_ts3.23.1901.2014.tmp.dat.nc', 'r') as temp_data,\
     netCDF4.Dataset('data/cru_ts3.23.1901.2014.pre.dat.nc', 'r') as pre_data,\
     netCDF4.Dataset('data/cru_ts3.23.1901.2014.cld.dat.nc', 'r') as cld_data:

    lat = numpy.array(list(temp_data.variables['lat']))
    lon = numpy.array(list(temp_data.variables['lon']))

    temp = temp_data.variables['tmp']
    pre = pre_data.variables['pre']
    cld = cld_data.variables['cld']

    days = numpy.array(list(temp_data.variables['time']))
    start = datetime.date(1900, 1, 1)  # dates are measured from here
    years = list(map(lambda d: (start + datetime.timedelta(days=d)).year, days))

    for glacier in names:
        glacier_lat_index = closest_index_in_range(-89.75, 89.75, 0.5,
                                                   glaciers.loc[glacier, 'lat'])
        glacier_lon_index = closest_index_in_range(-179.75, 179.75, 0.5,
                                                   glaciers.loc[glacier, 'lon'])

        grid_square_temps = temp[:, glacier_lat_index, glacier_lon_index].reshape([len(years)//12,
                                                                                   12])

        # Check for at least NUM_YEARS years of data
        if grid_square_temps.shape[0] >= NUM_YEARS:
            # TODO: check if correct
            continentality = (grid_square_temps[-NUM_YEARS:, :].max(axis=1).mean() -
                              grid_square_temps[-NUM_YEARS:, :].min(axis=1).mean())

            glaciers.loc[glacier, 'continentality'] = continentality

            if glaciers.loc[glacier, 'lat'] > 0:
                # Northern Hemisphere, use June, July, August
                mean_summer_temp = grid_square_temps[-NUM_YEARS:, 5:8].mean()
            else:
                # Southern Hemisphere, use December, January, February
                mean_summer_temp = grid_square_temps[-NUM_YEARS:, [0, 1, 11]].mean()
            glaciers.loc[glacier, 'summer_temperature'] = mean_summer_temp

        grid_square_pre = pre[:, glacier_lat_index, glacier_lon_index].reshape([len(years)//12,
                                                                                12])

        # Check for at least NUM_YEARS years of data
        if grid_square_pre.shape[0] >= NUM_YEARS:
            mean_pre = grid_square_pre[-NUM_YEARS:, :].sum(axis=1).mean()
            glaciers.loc[glacier, 'precipitation'] = mean_pre

            if glaciers.loc[glacier, 'lat'] > 0:
                # Northern Hemisphere, use December, January, February
                mean_winter_pre = grid_square_pre[-NUM_YEARS:, [0, 1, 11]].sum(axis=1).mean()
            else:
                # Southern Hemisphere, use June, July, August
                mean_winter_pre = grid_square_pre[-NUM_YEARS:, 5:8].sum(axis=1).mean()
            glaciers.loc[glacier, 'winter_precipitation'] = mean_winter_pre

        grid_square_cld = cld[:, glacier_lat_index, glacier_lon_index].reshape([len(years)//12,
                                                                                12])

        # Check for at least NUM_YEARS years of data
        if grid_square_cld.shape[0] >= NUM_YEARS:
            mean_cld = grid_square_cld[-NUM_YEARS:, :].mean()
            glaciers.loc[glacier, 'cloud_cover'] = mean_cld

with netCDF4.Dataset('data/air.mon.mean.nc', 'r') as lapse_temp,\
     netCDF4.Dataset('data/hgt.mon.mean.nc', 'r') as lapse_height:

    temp_var = lapse_temp.variables['air'][-(15 + NUM_YEARS*12):-15, :, :, :]
    height_var = lapse_height.variables['hgt'][-(15 + NUM_YEARS*12):-15, :, :, :]

    # Lapse rates
    for glacier in names:
        glacier_lat_index = 72 - closest_index_in_range(-90, 90, 2.5, glaciers.loc[glacier, 'lat'])
        lon = glaciers.loc[glacier, 'lon']

        # conversion between 0 to 360 and -180 to 180 degree system
        glacier_lon_index = (closest_index_in_range(-180, 180, 2.5,
                                                    lon if lon < 180
                                                    else -(180 - (lon - 180))) + 72) % 144

        grid_square_temps = temp_var[:, :, glacier_lat_index, glacier_lon_index]
        grid_square_heights = height_var[:, :, glacier_lat_index, glacier_lon_index]

        lapse_rates = []

        for i in range(NUM_YEARS*12):
            heights = grid_square_heights[i, :]
            temps = grid_square_temps[i, :]

            index = heights < 10000  # restrict to troposphere
            lapse_rates.append(-linregress(heights[index], temps[index])[0])

        lapse_mean = numpy.array(lapse_rates).reshape(NUM_YEARS, 12).mean(axis=1).mean()

        glaciers.loc[glacier, 'lapse_rate'] = lapse_mean

glaciers.to_pickle('data/serialized/glaciers_climate')
