import pandas
import numpy
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
import scipy.stats

from data import p, gamma, RGI_REGIONS

single_data = pandas.read_pickle('data/serialized/single_data')
single_data = single_data[-(single_data['sensitivity']/single_data['volumes_ss']) < 1]

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

sensitivity = importr('sensitivity')

bif_dists = []
sensitivities = []
for region in RGI_REGIONS:
    bif_dist = single_data.loc[region]['bif_dist']
    vols = single_data.loc[region]['volumes_ss']
    sens = single_data.loc[region]['sensitivity']
    mask = vols > 0
    bif_dists.append(sum(vols[mask]*bif_dist[mask])/sum(vols[mask]))
    sensitivities.append(sum(-sens[mask])/sum(vols[mask]))
bif_dists = pandas.DataFrame(bif_dists, index=RGI_REGIONS)
sensitivities = pandas.DataFrame(sensitivities, index=RGI_REGIONS)
tau = numpy.exp(numpy.log(single_data['tau']).groupby(level='Region').mean())

cl = all_glaciers['volume']/all_glaciers['length']**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma
means = pandas.concat([numpy.exp(numpy.log(all_glaciers['volume']).groupby(level='Region').mean()),
                       numpy.exp(numpy.log(cl).groupby(level='Region').mean()),
                       numpy.exp(numpy.log(ca).groupby(level='Region').mean()),
                       all_glaciers['G'].groupby(level='Region').mean(),
                       all_glaciers['g_abl'].groupby(level='Region').mean(),
                       numpy.arctan(all_glaciers['SLOPE_avg']).groupby(level='Region').mean(),
                       bif_dists],
                      axis=1)
means.columns = ['volume', 'cl', 'ca', 'G', 'g_abl', 'slope', 'bif_dist']

for var in means.columns:
    print('tau', var, scipy.stats.pearsonr(means[var].loc[RGI_REGIONS], tau.loc[RGI_REGIONS]))
    print('sensitivity', var, scipy.stats.pearsonr(means[var].loc[RGI_REGIONS],
                                                   sensitivities[0].loc[RGI_REGIONS]))
