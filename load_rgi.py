import pickle
import re

import numpy
import pandas
import geopandas
import geopy

# TODO: what to do with Alaska?

regions_thickness = ['alaska', 'westerncanada', 'arcticcanadaN',
                     'arcticcanadaS', 'greenland', 'iceland',
                     'svalbard', 'scandinavia', 'russianarctic',
                     'northasia', 'centraleurope', 'caucasus',
                     'centralasiaN', 'centralasiaW', 'centralasiaS',
                     'lowlatitudes', 'southernandes', 'newzealand',
                     'antarctic']

regions_rgi = ['Alaska', 'WesternCanadaUS','ArcticCanadaNorth',
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic']

data = pandas.DataFrame(columns=['RGIId', 'Slope', 'Aspect'])

thickness_re = re.compile('^(\d+);' + '\s+([-.0-9]+)'*18 + ';\s+(\d+)\s+(.+)$', flags=re.MULTILINE)

all_regions = []

for i, region in enumerate(regions_rgi):
    region_data = geopandas.read_file('{num}_rgi50_{name}'.format(num=str(i + 1).zfill(2), name=region))

    data = data.append(region_data[['RGIId', 'Slope', 'Aspect']])

    region_data = region_data.drop(['BgnDate', 'EndDate', 'GLIMSId', 'geometry'], axis=1)
    region_data['Region'] = region
    region_data.set_index(['RGIId'])

    thick_file = open('thick/thick_{name}_0.00_999.00.dat'.format(name=regions_thickness[i]), 'r').read()

    thick_data = pandas.DataFrame(thickness_re.findall(thick_file), columns=['ID', 'Location_x', 'Location_y', 'area', 'volume', 'THICK_mean', 'THICK_max', 'ELEV_min', 'ELEV_max', 'ELEV_med', 'LENGTH', 'SLOPE_avg', 'SLOPE_band', 'TAU_avg', 'TAU_min', 'TAU_max', 'SF_avg', 'SF_min', 'R_V', 'Survey_year', 'Name'])

    thick_data = thick_data.set_index('ID')

    thick_data = thick_data.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

    region_data['RGIId'] = region_data['RGIId'].str[-5:]
    region_data = region_data.set_index('RGIId')

    region_data['Thickness'] = thick_data['THICK_mean']/1000
    region_data[['Location_x', 'Location_y', 'area', 'volume', 'THICK_mean', 'THICK_max', 'ELEV_min', 'ELEV_max', 'ELEV_med', 'LENGTH', 'SLOPE_avg', 'SLOPE_band']] = thick_data[['Location_x', 'Location_y', 'area', 'volume', 'THICK_mean', 'THICK_max', 'ELEV_min', 'ELEV_max', 'ELEV_med', 'LENGTH', 'SLOPE_avg', 'SLOPE_band']]

    hypso_data = pandas.read_csv('{num}_rgi50_{name}/{num}_rgi50_{name}_hypso.csv'.format(num=str(i + 1).zfill(2), name=region))

    hypso_data['RGIId   '] = hypso_data['RGIId   '].str[-5:]
    hypso_data = hypso_data.set_index('RGIId   ')
    
    altitudes = numpy.array(list(map(numpy.float, hypso_data.columns[2:])))

    # hypsometric maximum
    region_data['ELA'] = altitudes[numpy.argmax(hypso_data.values[:, 2:], axis=1)]

    # remove glaciers with no hypsometry data (-9 in RGI)
    region_data = region_data[~(hypso_data.values[:, 2:] < 0).any(axis=1)]

    # remove glaciers with unknown slope (-9 in RGI)
    region_data = region_data[region_data['Slope'] > 0]

    # convert to radians
    region_data['Slope'] *= numpy.pi/180

    # restrict to glacier type
    region_data = region_data[region_data['GlacType'].str[0] == '0']

    # remove tidewater glaciers, ones that have minimum altitude of 0
    region_data = region_data[region_data['Zmin'] > 0]

    # remove glaciers with quantities of 0
    region_data = region_data[region_data['volume'] > 0]
    region_data = region_data[region_data['LENGTH'] > 0]
    region_data = region_data[region_data['SLOPE_avg'] > 0]

    region_data = region_data.reset_index()
    region_data = region_data.set_index(['Region', 'RGIId'])

    all_regions.append(region_data)

all_glaciers = pandas.concat(all_regions).dropna()
all_glaciers = all_glaciers.drop('LowLatitudes')

pickle.dump(all_glaciers, open('all_glaciers.p', 'wb'))

# Load slope and aspect data into WGMS data
data = data.set_index(['RGIId'])

data_id = pandas.read_csv('MAIL_WGMS/00_rgi50_links.20151130_WithCategories.csv',
                          usecols=['RGIId', 'FoGId'],
                          index_col=['FoGId'],
                          encoding='ISO-8859-1')

glaciers = pickle.load(open('glaciers', 'br'))

conv = data_id.loc[glaciers.index.values].dropna()

rgi_ids = list(conv.values.flatten())

conv['Slope'] = data.loc[rgi_ids]['Slope'].values
conv['Aspect'] = data.loc[rgi_ids]['Aspect'].values
glaciers[['slope', 'aspect']] = conv[['Slope', 'Aspect']]
