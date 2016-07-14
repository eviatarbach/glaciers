import pandas
import numpy
from scipy.stats import linregress

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
    glaciers_ela = pandas.DataFrame({'WGMS_ID': names, 'year': numpy.nan})
    glaciers_ela = glaciers_ela.set_index(['WGMS_ID', 'year'])

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
            altitudes = data_mb.loc[glacier, year]['ALTITUDE'].values
            balance = data_mb.loc[glacier, year]['ANNUAL_BALANCE'].values

            g = linregress(altitudes, balance).slope
            gradients.append(g)

            ela = data_mb.loc[glacier, year]['ela'].iloc[0]
            ela_i = altitudes.searchsorted(ela)

            glaciers_ela.loc[(glacier, year), 'ela'] = ela

            # At least 4 mass balance measurements on either side of
            # the ELA
            if (ela_i >= 4) and (len(balance) - ela_i >= 4):
                g_abl = linregress(altitudes[:ela_i], balance[:ela_i]).slope
                g_acc = linregress(altitudes[ela_i:], balance[ela_i:]).slope

                gradients_abl.append(g_abl)
                gradients_acc.append(g_acc)

        glaciers.loc[glacier, 'g'] = numpy.mean(gradients)/1000
        glaciers.loc[glacier, 'g_abl'] = numpy.mean(gradients_abl)/1000
        glaciers.loc[glacier, 'g_acc'] = numpy.mean(gradients_acc)/1000
        glaciers.loc[glacier, 'g_abl_std'] = numpy.std(gradients_abl)/1000
        glaciers.loc[glacier, 'g_acc_std'] = numpy.std(gradients_acc)/1000

glaciers_ela.index = glaciers_ela.index.set_levels([glaciers_ela.index.levels[0],
                                                    glaciers_ela.index.levels[1].astype(int)])
glaciers_ela = glaciers_ela.dropna()

glaciers.to_pickle('data/serialized/glaciers_climate')
glaciers_ela.to_pickle('data/serialized/glaciers_ela')
