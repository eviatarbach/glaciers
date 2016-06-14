import pickle
import re

import numpy
import pandas
import geopandas

from data import RGI_REGIONS, THICK_REGIONS

# TODO: what to do with Alaska?


def median_elev(hypso, elevs):
    """
    Find the altitude of median hypsometry (the altitude at which the
    sum of the area bands is half of the total). Since the area bands
    are discrete, linear interpolation is used to approximate the
    altitude.
    """
    cumsum = numpy.cumsum(hypso, axis=1)

    # The sum of the area bands is normalized to 1000, so the median
    # altitude is the altitude at which the sum of the bands below it
    # is 500.

    # Indices that straddle the median
    i_upper = numpy.apply_along_axis(lambda a: a.searchsorted(500), axis=1, arr=cumsum)
    i_lower = i_upper - 1

    cumsum_lower = cumsum.values[numpy.arange(len(cumsum)), i_lower]
    cumsum_upper = cumsum.values[numpy.arange(len(cumsum)), i_upper]

    # Linear interpolation to approximate median elevation: set median
    # elevation such that
    # (hypso_median - hypso_lower)/(hypso_upper - hypso_lower)
    # = (elev_median - elev_lower)/(elev_upper - elev_lower). The
    # hypsometry format guarantees that elev_upper - elev_lower = 50
    # and hypso_median = 500.
    return elevs[i_lower] + (50*(500 - cumsum_lower)/(cumsum_upper - cumsum_lower))

data = pandas.DataFrame(columns=['RGIId', 'Slope', 'Aspect'])

thickness_re = re.compile('^(\d+);' + '\s+([-.0-9]+)'*18 + ';\s+(\d+)\s+(.+)$', flags=re.MULTILINE)

all_regions = []

for i, region in enumerate(RGI_REGIONS):
    region_data = geopandas.read_file('data/{num}_rgi50_{name}'.format(num=str(i + 1).zfill(2),
                                                                       name=region))

    data = data.append(region_data[['RGIId', 'Slope', 'Aspect']])

    region_data = region_data.drop(['BgnDate', 'EndDate', 'GLIMSId', 'geometry'], axis=1)
    region_data['Region'] = region
    region_data.set_index(['RGIId'])

    thick_file = open('data/thick/thick_{name}_0.00_999.00.dat'.format(name=THICK_REGIONS[i]),
                      'r').read()

    thick_data = pandas.DataFrame(thickness_re.findall(thick_file),
                                  columns=['ID', 'Location_x', 'Location_y', 'area', 'volume',
                                           'THICK_mean', 'THICK_max', 'ELEV_min', 'ELEV_max',
                                           'ELEV_med', 'LENGTH', 'SLOPE_avg', 'SLOPE_band',
                                           'TAU_avg', 'TAU_min', 'TAU_max', 'SF_avg', 'SF_min',
                                           'R_V', 'Survey_year', 'Name'])

    thick_data = thick_data.set_index('ID')

    thick_data = thick_data.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

    region_data['RGIId'] = region_data['RGIId'].str[-5:]
    region_data = region_data.set_index('RGIId')

    region_data['Thickness'] = thick_data['THICK_mean']/1000

    values = ['Location_x', 'Location_y', 'area', 'volume', 'THICK_mean', 'THICK_max', 'ELEV_min',
              'ELEV_max', 'ELEV_med', 'LENGTH', 'SLOPE_avg', 'SLOPE_band']
    region_data[values] = thick_data[values]

    hypso_data = pandas.read_csv('data/{num}_rgi50_{name}/{num}_rgi50_{name}_hypso.csv'
                                 .format(num=str(i + 1).zfill(2), name=region))

    hypso_data['RGIId   '] = hypso_data['RGIId   '].str[-5:]
    hypso_data = hypso_data.set_index('RGIId   ')

    # remove glaciers with no hypsometry data (-9 in RGI)
    region_data = region_data[~(hypso_data.values[:, 2:] < 0).any(axis=1)]
    hypso_data = hypso_data[~(hypso_data.values[:, 2:] < 0).any(axis=1)]

    altitudes = numpy.array(list(map(numpy.float, hypso_data.columns[2:])))

    # area-weighted mean altitude
    region_data['ELA'] = (altitudes*hypso_data[hypso_data.columns[2:]]/1000).sum(axis=1)

    # median altitude
    region_data['ELA2'] = median_elev(hypso_data[hypso_data.columns[2:]], altitudes)

    # region_data['ELA'] = altitudes[numpy.argmax(hypso_data.values[:, 2:], axis=1)]

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
# all_glaciers = all_glaciers.drop('LowLatitudes')

pickle.dump(all_glaciers, open('data/serialized/all_glaciers', 'wb'))

# Load slope and aspect data into WGMS data
data = data.set_index(['RGIId'])

data_id = pandas.read_csv('data/MAIL_WGMS/00_rgi50_links.20151130_WithCategories.csv',
                          usecols=['RGIId', 'FoGId'], index_col=['FoGId'], encoding='ISO-8859-1')

glaciers = pickle.load(open('data/serialized/glaciers_climate', 'br'))

conv = data_id.loc[glaciers.index.values].dropna()

rgi_ids = list(conv.values.flatten())

conv['Slope'] = data.loc[rgi_ids]['Slope'].values
conv['Aspect'] = data.loc[rgi_ids]['Aspect'].values
glaciers[['slope', 'aspect']] = conv[['Slope', 'Aspect']]
