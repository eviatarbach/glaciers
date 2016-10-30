import multiprocessing

import numpy
import pandas

from matlab_to_python import loadmat

from data import p, gamma, closest_index_in_range, RGI_REGIONS

data = loadmat('data/glaciers/dT_GISS_rcp45_monthly_2006_2100n.mat')

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

sensitivity_ELA = 89
dt = 1/12

date_range = pandas.date_range('1/1/2006', '12/31/2100', freq='M')


def process_glacier(glacier):
    #if glacier[0] in ['AntarcticSubantarctic', 'Alaska', 'SouthernAndes']:
    #    return
    glacier_data = all_glaciers.loc[glacier]
    if glacier_data.isnull().any():
        return numpy.zeros(1128)
    V = glacier_data['volume']
    L = glacier_data['LENGTH']
    H = glacier_data['THICK_mean']
    A = glacier_data['area']
    beta = glacier_data['SLOPE_avg']
    if pandas.isnan(beta):
        beta = glacier_data['Slope']
    Zmax = glacier_data['Zmax']

    cl = V/(L**p)
    ca = V/(A**gamma)

    ela = glacier_data[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean()
    zela = ela - (Zmax - H)

    g_abl = glacier_data['g_abl']
    g_acc = glacier_data['g_acc']

    lat = glacier_data['lat']
    lon = glacier_data['lon']
    lat_index = closest_index_in_range(-90, 90, 0.5, lat)
    lon_index = closest_index_in_range(0, 359.5, 0.5, lon + 180)

    dT = data['dT'][numpy.ravel_multi_index((lon_index, lat_index), (720, 361))]/100
    series = pandas.Series(dT, date_range).rolling(12, center=True).mean()
    series = series.dropna().values

    volumes = numpy.zeros(len(series) - 1)
    for step in range(1, len(series)):
        ela += (series[step] - series[step - 1])*sensitivity_ELA
        zela = ela - (Zmax - H)
        H = ca**(1/gamma)*V**(1 - 1/gamma)
        W = ca**(-1/gamma)*cl**(1/p)*V**(1/gamma - 1/p)
        L = cl**(-1/p)*V**(1/p)
        dV = dt*((g_acc - g_abl)*(H - zela)**2*W/(2*beta)
                 + g_abl*((H - zela)*W*L - beta*W*L**2/2))
        V += dV
        if V <= 0:
            V = 0
            volumes[step - 1] = V
            break
        volumes[step - 1] = V
    return volumes


def process_region(region):
    region_vols = numpy.zeros(1128)
    for glacier in all_glaciers.loc[region].index:
        vols = process_glacier((region, glacier))
        if not numpy.isnan(vols).any():
            region_vols += vols
    return region_vols

# pool = multiprocessing.Pool(processes=4)
# all_volumes = pool.map(process_region, RGI_REGIONS)
