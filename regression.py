import csv
import datetime
import bisect

import numpy
import scipy.io.netcdf as netcdf

glaciers = {}

with open('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-EE-MASS-BALANCE.csv') as f:
    reader = csv.reader(f)
    reader.next()  # skip first row of labels
    for row in reader:
        try:
            name = row[1]
            GLIMS_ID = int(row[2])
            year = int(row[3])
            lower_bound = float(row[4])
            upper_bound = float(row[5])
            if lower_bound == 9999:
                continue
            balance = float(row[11])
            if name not in glaciers.keys():
                glaciers[name] = {'id': GLIMS_ID, 'mass balance': {}}
            glacier = glaciers[name]
            if year not in glacier['mass balance'].keys():
                glacier['mass balance'][year] = {'data': []}
            glacier['mass balance'][year]['data'].append(((lower_bound + upper_bound)/2, balance))
        except ValueError:
            continue

for name in glaciers.keys():
    glacier = glaciers[name]
    for year in glacier['mass balance'].keys():
        array = numpy.array(glacier['mass balance'][year]['data'])
        altitudes = numpy.hstack([numpy.ones([array.shape[0], 1]), array[:, 0:1]])
        balance = array[:, 1:]
        b, m = numpy.linalg.lstsq(altitudes, balance)[0]
        glacier['mass balance'][year]['zela'] = -b/m
        glacier['mass balance'][year]['gradient'] = m

with open('DOI-WGMS-FoG-2015-11/WGMS-FoG-2015-11-A-GENERAL-INFORMATION.csv') as f:
    reader = csv.reader(f)
    reader.next()  # skip first row of labels
    for row in reader:
        name = row[1]
        if name in glaciers.keys():
            lat = float(row[9])
            lon = float(row[10])
            glaciers[name]['lat'] = lat
            glaciers[name]['lon'] = lon

temp_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.tmp.dat.nc', 'r')
pre_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.pre.dat.nc', 'r')

lat = numpy.array(list(temp_data.variables['lat']))
lon = numpy.array(list(temp_data.variables['lon']))
temp = temp_data.variables['tmp']

pre = pre_data.variables['pre']

days = numpy.array(list(temp_data.variables['time']))
start = datetime.date(1900, 1, 1)  # dates are measured from here
years = map(lambda d: (start + datetime.timedelta(days=d)).year, days)

for glacier in glaciers.values():
    glacier_lat_index = bisect.bisect(lat, glacier['lat'])
    glacier_lon_index = bisect.bisect(lon, glacier['lon'])

    grid_square_temps = temp[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])
    temp_range = grid_square_temps.max(axis=1) - grid_square_temps.min(axis=1)

    grid_square_pre = pre[:, glacier_lat_index, glacier_lon_index].reshape([len(years)/12, 12])
    total_pre = grid_square_pre.sum(axis=1)

    # TODO: fix this bug
    if (total_pre > 1e20).any():
        continue

    for i, annual_range in enumerate(temp_range):
        year = 1900 + i
        if year in glacier['mass balance'].keys():
            glacier['mass balance'][year]['temperature range'] = annual_range
            glacier['mass balance'][year]['precipitation'] = total_pre[i]
    #glacier['temperature range'] = temp_range
    #glacier['precipitation'] = total_pre

#num_years = sum(map(lambda g: len(g['mass balance']), glaciers.values()))

#X = numpy.zeros([num_years, 2])
#y = numpy.zeros([num_years, 1])

ranges = []
precipitation = []
zela = []

for glacier in glaciers.values():
    #temperature_range = glacier['temperature range']
    #precipitation = glacier['precipitation']
    for year in glacier['mass balance']:
        try:
            if glacier['mass balance'][year]['zela'] < -100000:
                continue
            ranges.append(glacier['mass balance'][year]['temperature range'])
            precipitation.append(glacier['mass balance'][year]['precipitation'])
            zela.append(glacier['mass balance'][year]['zela'])
        except KeyError:
            pass

X = numpy.hstack([numpy.ones([len(zela), 1]), numpy.array([ranges]).transpose(), numpy.array([precipitation]).transpose()])
y = numpy.array(zela)
