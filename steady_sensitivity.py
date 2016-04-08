import pickle

import numpy
import matplotlib.pyplot as plt
import mpmath
import scipy
import scipy.interpolate
import scipy.optimize

from matlab_to_python import loadmat

RGI_REGIONS = ['Alaska', 'WesternCanadaUS','ArcticCanadaNorth',
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic']

RGI_NAMES = ['Alaska', 'Western Canada and USA','Arctic Canada (North)',
             'Arctic Canada (South)', 'Greenland (periphery)', 'Iceland',
             'Svalbard and Jan Mayen', 'Scandinavia', 'Russian Arctic', 'North Asia',
             'Central Europe', 'Caucasus and Middle East', 'Central Asia',
             'South Asia (West)', 'South Asia (East)', 'Low Latitudes',
             'Southern Andes', 'New Zealand', 'Antarctic and Subantarctic']

#f = pickle.load(open('interpolant.p', 'rb'))

#f = numpy.vectorize(f)

all_glaciers = pickle.load(open('all_glaciers.p', 'rb'))

P = pickle.load(open('P.p', 'rb'))
V = pickle.load(open('V.p', 'rb'))
P2 = pickle.load(open('P2.p', 'rb'))
V2 = pickle.load(open('V2.p', 'rb'))

f1 = scipy.interpolate.interp1d(P, V)
f2 = scipy.interpolate.interp1d(P2, V2)

alpha = 8/11
beta = 13/11
def F(P, V): return -V**(alpha)*P - V**(beta) + V

def f(P):
    if P > 0.1859:
        return numpy.float64(0)
    elif P > -2.526:
        return numpy.float64(f2(P))
    elif P > -300.01:
        return numpy.float64(f1(P))
    else:
        roots = numpy.roots([-1, 0, 1, 0, 0, -P, 0, 0, 0, 0, 0, 0, 0, 0])
        return (roots[numpy.nonzero(numpy.logical_and(numpy.isreal(roots), roots != 0))].real**11)[0]

def Pval(V):
    if V < 0.0612:
        return float('inf')
    return scipy.optimize.minimize(lambda P: (f(P) - V)**2, 0, bounds=[(None, 0.1859)]).x

Pval_vec = numpy.vectorize(Pval)
f_vec = numpy.vectorize(f)

a = 0.6
q = 2.2
gamma = 1.375

all_P = numpy.array([])
all_L = []
all_V = []

region_volumes = []

sensitivities = []

diff = numpy.vectorize(lambda p: mpmath.diff(lambda p2: f_vec(p2), p, h=1e-6))

for i, region_name in enumerate(RGI_REGIONS):
    print(region_name)
    region = all_glaciers.loc[region_name]

    if len(region):
        volumes = region['volume']
        heights = region['Thickness']
        lengths = region['LENGTH']
        slopes = region['SLOPE_avg']*numpy.pi/180
        areas = region['area']

        cl = volumes/(lengths**q)
        ca = volumes/(areas**gamma)
        cw = (ca/cl)**(1/(q - gamma))

        Ldim = ((2*cl**((a + 2)/q)*cw**a)/(slopes*numpy.cos(slopes)))**(q/(3*(a - q + 2)))

        volumes_nd = volumes/Ldim**3

        P = Pval_vec(volumes_nd)

        sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/(slopes*numpy.cos(slopes))*diff(P)

        region_volume = sum(volumes.values[(P != float('inf')).nonzero()])

        sensitivities.append(sensitivity)
        region_volumes.append(region_volume)
