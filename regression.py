import csv
import datetime

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

temp_data = netcdf.netcdf_file('cru_ts3.23.1901.2014.tmp.dat.nc', 'r')

lat = numpy.array(list(temp_data.variables['lat']))
lon = numpy.array(list(temp_data.variables['lon']))
temp = numpy.array(list(temp_data.variables['lon']))

days = numpy.array(list(temp_data.variables['time']))
start = datetime.date(1900, 1, 1)  # dates are measured from here
years = map(lambda d: (start + datetime.timedelta(days=d)).year, days)
