import multiprocessing

import numpy
import pandas
from scipy.signal import butter, lfilter

from matlab_to_python import loadmat

from data import p, gamma, closest_index_in_range, RGI_REGIONS

GCMs = [('BCC', 3), ('CanESM', 1), ('CCSM', 1), ('CNRM', 3), ('CSIRO', 1), ('GFDL', 1),
        ('GISS', 3), ('HadGEM', 1), ('INM', 1), ('IPSL', 3), ('MIROC', 1), ('MPIESM', 3),
        ('MRI', 1), ('Nor', 1)]

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2

sensitivity_ELA = 89
dt = 1

date_ranges = (pandas.date_range('1/1/2006', '12/31/2100', freq='M'),
               pandas.date_range('1/1/2101', '12/31/2200', freq='M'),
               pandas.date_range('1/1/2101', '12/31/2200', freq='M'))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def process_glacier(glacier):
    glacier_data = all_glaciers.loc[glacier]
    V = glacier_data['volume']
    L = glacier_data['Lmax']
    H = glacier_data['THICK_mean']
    A = glacier_data['area']
    beta = glacier_data['SLOPE_avg']
    Zmax = glacier_data['Zmax']

    cl = V/(L**p)
    ca = V/(A**gamma)

    ela_0 = (glacier_data[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean() + Zmax)/2

    g_abl = glacier_data['g_abl']
    g_acc = glacier_data['g_acc']

    lat = glacier_data['lat']
    lon = glacier_data['lon']
    lat_index = closest_index_in_range(-90, 90, 0.5, lat)
    lon_index = closest_index_in_range(0, 359.5, 0.5, lon + 180)

    volumes_full = []
    for i, data_years in enumerate(data):
        dT = data_years[numpy.ravel_multi_index((lon_index, lat_index), (720, 361))]/100
        series = butter_bandpass_filter(dT.reshape(-1, 12).mean(axis=1), 0, 1/50., 1, order=4)
        # series = pandas.Series(dT.reshape(-1, 12).mean(axis=1)).rolling(20, center=True).mean()
        # series = series.dropna().values
        series = series - series[0]

        volumes = numpy.zeros(len(series))
        for step in range(len(series)):
            ela = ela_0 + series[step]*sensitivity_ELA
            zela = ela - (Zmax - H)

            H = ca**(1/gamma)*V**(1 - 1/gamma)
            W = ca**(-1/gamma)*cl**(1/p)*V**(1/gamma - 1/p)
            L = cl**(-1/p)*V**(1/p)
            dV = dt*((g_acc - g_abl)*(H - zela)**2*W/(2*beta)
                     + g_abl*((H - zela)*W*L - beta*W*L**2/2))
            V += dV
            if V <= 0:
                V = 0
                volumes[step] = V
                break
            volumes[step] = V
        volumes_full.append(volumes)

    return numpy.concatenate(volumes_full)


def process_region(region):
    print(region)
    region_vols = numpy.zeros(95 + 100*(len(data) - 1))
    for glacier in all_glaciers.loc[region].index:
        vols = process_glacier((region, glacier))
        if not numpy.isnan(vols).any():
            region_vols += vols
    return region_vols

all_volumes = dict([('BCC', {}), ('CanESM', {}), ('CCSM', {}), ('CNRM', {}), ('CSIRO', {}),
                    ('GFDL', {}), ('GISS', {}), ('HadGEM', {}), ('INM', {}), ('IPSL', {}),
                    ('MIROC', {}), ('MPIESM', {}), ('MRI', {}), ('Nor', {})])

for GCM in GCMs[:1]:
    for rcp in ['45', '85'][:1]:
        year_ranges = ['2006_2100', '2101_2200', '2201_2300'][:1] #[:GCM[1]]
        data = []
        for year_range in year_ranges:
            mat = loadmat('data/glaciers'
                          '/dT_{GCM}_rcp{rcp}_monthly_{years}n.mat'.format(GCM=GCM[0],
                                                                           rcp=rcp,
                                                                           years=year_range))
            data.append(mat['dT'])
        pool = multiprocessing.Pool(processes=4)
        volumes = pool.map(process_region, RGI_REGIONS)
        pool.close()
        all_volumes[GCM[0]][rcp] = volumes
