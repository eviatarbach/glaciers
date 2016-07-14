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

thickness_regex = re.compile('^(\d+);' + '\s+([-.0-9]+)'*18 + ';\s+(\d+)\s+(.+)$',
                             flags=re.MULTILINE)

all_regions = []

for i, region in enumerate(RGI_REGIONS):
    region_id = str(i + 1).zfill(2)
    with open('data/thick/thick_{name}_0.00_999.00.dat'.format(name=THICK_REGIONS[i]),
              'r') as thick_file,\
         open('data/{num}_rgi50_{name}/{num}_rgi50_{name}_hypso.csv'.format(num=region_id,
                                                                            name=region),
              'r') as hypso_file:

        # No context managers in geopandas?
        region_data = geopandas.read_file('data/{num}_rgi50_{name}'.format(num=region_id,
                                                                           name=region))
        data = data.append(region_data[['RGIId', 'Slope', 'Aspect']])

        region_data = region_data.drop(['BgnDate', 'EndDate', 'GLIMSId', 'geometry'], axis=1)
        region_data['Region'] = region
        region_data.set_index(['RGIId'])

        region_data = region_data.rename(columns={'CenLat': 'lat', 'CenLon': 'lon'})

        thick_data = pandas.DataFrame.from_records(thickness_regex.findall(thick_file.read()),
                                                   columns=['ID', 'Location_x', 'Location_y',
                                                            'area', 'volume', 'THICK_mean',
                                                            'THICK_max', 'ELEV_min', 'ELEV_max',
                                                            'ELEV_med', 'LENGTH', 'SLOPE_avg',
                                                            'SLOPE_band', 'TAU_avg', 'TAU_min',
                                                            'TAU_max', 'SF_avg', 'SF_min', 'R_V',
                                                            'Survey_year', 'Name'],
                                                   exclude=['THICK_max', 'SLOPE_band', 'TAU_avg',
                                                            'TAU_min', 'TAU_max', 'SF_avg',
                                                            'SF_min', 'R_V', 'Survey_year',
                                                            'Name'])

        thick_data = thick_data.set_index('ID')

        thick_data = thick_data.apply(pandas.to_numeric)

        region_data['RGIId'] = region_data['RGIId'].str[-5:]
        region_data = region_data.set_index('RGIId')

        column_names = ['Location_x', 'Location_y', 'area', 'volume', 'THICK_mean', 'ELEV_min',
                        'ELEV_max', 'ELEV_med', 'LENGTH', 'SLOPE_avg']
        region_data[column_names] = thick_data[column_names]

        # -9 indicates no hypsometry data
        hypso_data = pandas.read_csv(hypso_file, na_values='  -9')

        hypso_data['RGIId   '] = hypso_data['RGIId   '].str[-5:]
        hypso_data = hypso_data.set_index('RGIId   ')

        # region_data = region_data[~(hypso_data.values[:, 2:] == -9).any(axis=1)]
        # hypso_data = hypso_data[~(hypso_data.values[:, 2:] == -9).any(axis=1)]

        altitudes = numpy.float64(hypso_data.columns[2:])

        # area-weighted mean altitude
        region_data['ELA_weighted'] = (altitudes*hypso_data[hypso_data.columns[2:]]
                                       /1000).sum(axis=1)

        # median altitude
        region_data['ELA_median'] = median_elev(hypso_data[hypso_data.columns[2:]], altitudes)

        # region_data['ELA'] = altitudes[numpy.argmax(hypso_data.values[:, 2:], axis=1)]

        # replace invalid values with NaN
        region_data['volume'] = region_data['volume'].replace(0, numpy.nan)
        region_data['LENGTH'] = region_data['LENGTH'].replace(0, numpy.nan)
        region_data['SLOPE_avg'] = region_data['SLOPE_avg'].replace(0, numpy.nan)
        region_data['Slope'] = region_data['Slope'].replace(-9, numpy.nan)
        region_data['Lmax'] = region_data['Lmax'].replace(-9, numpy.nan)

        # unit conversion

        # degrees to radians
        region_data['SLOPE_avg'] *= numpy.pi/180
        region_data['Slope'] *= numpy.pi/180

        # km to m
        region_data['Area'] *= 1000**2
        region_data['area'] *= 1000**2
        region_data['volume'] *= 1000**3
        region_data['LENGTH'] *= 1000

        # restrict to glacier type
        region_data = region_data[region_data['GlacType'].str[0] == '0']

        # remove tidewater glaciers, ones that have minimum altitude of 0
        region_data = region_data[region_data['Zmin'] > 0]

        region_data = region_data.reset_index()
        region_data = region_data.set_index(['Region', 'RGIId'])

        all_regions.append(region_data)

all_glaciers = pandas.concat(all_regions)  # .dropna()
# all_glaciers = all_glaciers.drop('LowLatitudes')

with open('data/GlaThiDa_2014/T.csv', 'r', encoding='ISO-8859-1') as glathida_file,\
     open('data/Manual_links_GlaThiDa_to_RGI_WORLD_20160412.csv', 'r',
          encoding='ISO-8859-1') as glathida_to_rgi_file:
    glathida = pandas.read_csv(glathida_file, usecols=['GlaThiDa_ID', 'MEAN_THICKNESS'], sep=';',
                               index_col=['GlaThiDa_ID'], skiprows=[0, 1, 3]).dropna()

    glathida.index = glathida.index.astype(numpy.int64)

    glathida_to_rgi = pandas.read_csv(glathida_to_rgi_file, index_col=['GlaThiDa_ID'],
                                      usecols=['GlaThiDa_ID', 'RGI_ID', 'status'],
                                      encoding='ISO-8859-1')

    # Remove glaciers with status = False, indicates that mapping may not be correct
    glathida_to_rgi = glathida_to_rgi[glathida_to_rgi['status']].dropna()

    conv = glathida_to_rgi.loc[glathida.index].dropna()
    conv['MEAN_THICKNESS'] = glathida['MEAN_THICKNESS']

    rgi_regions = all_glaciers.index.levels[0][conv['RGI_ID'].str[6:8].astype(int)]
    rgi_ids = conv['RGI_ID'].str[9:].values

    indices = list(zip(rgi_regions, rgi_ids))
    valid_mask = numpy.nonzero([i in all_glaciers.index for i in indices])[0]
    valid_indices = [indices[i] for i in valid_mask]

    all_glaciers.loc[valid_indices, 'THICK_mean'] = conv['MEAN_THICKNESS'].values[valid_mask]

    #all_glaciers.to_pickle('data/serialized/all_glaciers')

# with open('data/MAIL_WGMS/00_rgi50_links.20151130_WithCategories.csv', 'r',
#           encoding='ISO-8859-1') as id_file:
#     # Load slope and aspect data into WGMS data
#     data = data.set_index(['RGIId'])
#
#     data_id = pandas.read_csv(id_file, usecols=['RGIId', 'FoGId'], index_col=['FoGId'])
#
#     glaciers = pandas.read_pickle('data/serialized/glaciers_climate')
#
#     conv = data_id.loc[glaciers.index.values].dropna()
#
#     rgi_ids = list(conv.values.flatten())
#
#     conv['Slope'] = data.loc[rgi_ids]['Slope'].values
#     conv['Aspect'] = data.loc[rgi_ids]['Aspect'].values
#     glaciers[['slope', 'aspect']] = conv[['Slope', 'Aspect']]