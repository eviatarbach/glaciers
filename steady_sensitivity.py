import pickle

import numpy
import matplotlib.pyplot as plt
import mpmath
import scipy
import scipy.interpolate
import scipy.optimize

from matlab_to_python import loadmat
from data import RGI_REGIONS, RGI_NAMES, a, q, gamma, f, f_vec, alpha, delta, diff, P0, V0

#f = pickle.load(open('interpolant.p', 'rb'))

#f = numpy.vectorize(f)

all_glaciers = pickle.load(open('all_glaciers.p', 'rb'))

def Pval(V):
    if V < V0:
        return float('inf')
    return scipy.optimize.minimize(lambda P: (f(P) - V)**2, 0, bounds=[(None, P0)]).x

Pval_vec = numpy.vectorize(Pval)

all_P = numpy.array([])
all_L = []
all_V = []

region_volumes = []

sensitivities = []

for i, region_name in enumerate(RGI_REGIONS):
    if region_name in ['AntarcticSubantarctic', 'Alaska']:
        continue

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

        region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

        sensitivities.append(sensitivity)
        region_volumes.append(region_volume)
