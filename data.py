import pickle

import numpy
import scipy
import scipy.interpolate
import mpmath

RGI_REGIONS = ['Alaska', 'AntarcticSubantarctic', 'ArcticCanadaNorth',
               'ArcticCanadaSouth', 'CaucasusMiddleEast', 'CentralAsia',
               'CentralEurope', 'GreenlandPeriphery', 'Iceland', 'LowLatitudes',
               'NewZealand', 'NorthAsia', 'RussianArctic', 'Scandinavia',
               'SouthAsiaEast', 'SouthAsiaWest', 'SouthernAndes', 'Svalbard',
               'WesternCanadaUS']

RGI_NAMES = ['Alaska', 'Antarctic and Subantarctic', 'Arctic Canada (North)',
             'Arctic Canada (South)', 'Caucasus and Middle East',
             'Central Asia', 'Central Europe', 'Greenland (periphery)',
             'Iceland', 'Low Latitudes', 'New Zealand', 'North Asia',
             'Russian Arctic', 'Scandinavia', 'South Asia (East)',
             'South Asia (West)', 'Southern Andes', 'Svalbard and Jan Mayen',
             'Western Canada and USA']

RGI_NAMES2 = ['Arctic Canada (North)',
             'Arctic Canada (South)', 'Caucasus and Middle East',
             'Central Asia', 'Central Europe', 'Greenland (periphery)',
             'Iceland', 'New Zealand', 'North Asia',
             'Russian Arctic', 'Scandinavia', 'South Asia (East)',
             'South Asia (West)', 'Southern Andes', 'Svalbard and Jan Mayen',
             'Western Canada and USA']

a = 1/3.
q = 5/3.
gamma = 1.25

P = pickle.load(open('P.p', 'rb'))
V = pickle.load(open('V.p', 'rb'))
P2 = pickle.load(open('P2.p', 'rb'))
V2 = pickle.load(open('V2.p', 'rb'))

f1 = scipy.interpolate.interp1d(P, V)
f2 = scipy.interpolate.interp1d(P2, V2)

alpha = 4/5.
delta = 7/5.

def F(P, V): return -V**(alpha)*P - V**(delta) + V

def f(P):
    if P > f2.x[-1]:
        return 0.0
    if f2.x[0] <= P <= f2.x[-1]:
        return f2(P)
    elif f1.x[0] <= P <= f1.x[-1]:
        return f1(P)
    else:
        roots = numpy.roots([-1, 0, 1, -P, 0, 0, 0, 0])
        return roots[numpy.nonzero(numpy.logical_and(numpy.isreal(roots), roots != 0))].real**5

f_vec = numpy.vectorize(f)

diff = numpy.vectorize(lambda p: float(mpmath.diff(lambda p2: f_vec(float(p2)), p, h=1e-6)))

P0 = 0.384900179459750
V0 = 0.06415

def latex_bars(names, values, scale):
    return '\n'.join(map(lambda n, l: n + ' & ' + '{:.3f}'.format(l) + ' & \mybar{' + str(scale) + '}{' + '{:.3f}'.format(l) + '}\\\\', names, values))
