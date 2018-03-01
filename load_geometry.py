import re

import numpy
import pandas
import geopandas

from data import RGI_REGIONS, THICK_REGIONS

thickness_regex = re.compile('^(\d+);' + '\s+([-.0-9]+)'*18 + ';\s+(\d+)\s+(.+)$',
                             flags=re.MULTILINE)

all_regions = []

total_num = []
total_area = []
included_num = []
included_area = []
for i, region in enumerate(RGI_REGIONS):
    region_id = str(i + 1).zfill(2)
    with open('data/thick/thick_{name}_0.00_999.00.dat'.format(name=THICK_REGIONS[i]),
              'r') as thick_file,\
         open('data/{num}_rgi50_{name}/{num}_rgi50_{name}_hypso.csv'.format(num=region_id,
                                                                            name=region),
              'r') as hypso_file:

        region_data = geopandas.read_file('data/{num}_rgi50_{name}'.format(num=region_id,
                                                                           name=region))

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

        altitudes = numpy.float64(hypso_data.columns[2:])
        areas = hypso_data[hypso_data.columns[2:]]

        # replace invalid values with NaN
        region_data['volume'] = region_data['volume'].replace(0, numpy.nan)
        region_data['THICK_mean'] = region_data['THICK_mean'].replace(0, numpy.nan)
        region_data['LENGTH'] = region_data['LENGTH'].replace(0, numpy.nan)
        region_data['SLOPE_avg'] = region_data['SLOPE_avg'].replace(0, numpy.nan)
        region_data['Slope'] = region_data['Slope'].replace(0, numpy.nan)
        region_data['Slope'] = region_data['Slope'].replace(-9, numpy.nan)
        region_data['Lmax'] = region_data['Lmax'].replace(-9, numpy.nan)

        # unit conversion

        # degrees to radians, then to slope
        region_data['SLOPE_avg'] = numpy.tan(region_data['SLOPE_avg']*numpy.pi/180)
        region_data['Slope'] = numpy.tan(region_data['Slope']*numpy.pi/180)

        # km to m
        region_data['Area'] *= 1000**2
        region_data['area'] *= 1000**2
        region_data['volume'] *= 1000**3
        region_data['LENGTH'] *= 1000

        print(region)
        print('Total number in RGI:', len(region_data))
        print('Total area in RGI (km^2):', region_data['Area'].sum()/1000**2)
        total_num.append(len(region_data))
        total_area.append(region_data['Area'].sum()/1000**2)

        # restrict to glacier type
        region_data = region_data[(region_data['GlacType'].str[0] == '0')
                                  & (region_data['GlacType'].str[1] == '0')
                                  & (region_data['RGIFlag'].str[0] == '0')]

        # remove tidewater glaciers, ones that have minimum altitude of 0
        region_data = region_data[region_data['Zmin'] > 0]

        # drop glaciers that have no slope information
        region_data = region_data[~region_data['SLOPE_avg'].isnull()
                                  | ~region_data['Slope'].isnull()]

        print('Total number included:', len(region_data))
        print('Total area included (km^2):', region_data['Area'].sum()/1000**2)
        included_num.append(len(region_data))
        included_area.append(region_data['Area'].sum()/1000**2)

        region_data = region_data.reset_index()
        region_data = region_data.set_index(['Region', 'RGIId'])

        all_regions.append(region_data)

all_glaciers = pandas.concat(all_regions)

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

    all_glaciers.to_pickle('data/serialized/all_glaciers')
