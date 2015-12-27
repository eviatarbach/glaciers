import math
import numpy
import mpmath
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
from matlab_to_python import loadmat

RGI_REGIONS = ['Alaska', 'WesternCanadaUS','ArcticCanadaNorth',
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic']

mat = loadmat('data.mat')['Regions']

region = mat['NorthAsia']

AAR = 0.6

volumes = region['volumes']
heights = region['heights']
lengths = region['lengths']
slopes = region['slopes']

zela = heights - AAR*lengths*numpy.tan(slopes)
P = 1 - zela/heights

hist, bins_height, bins_P = numpy.histogram2d(heights, P, bins=(50, 50), normed=True)

centred_height = (bins_height[:-1] + bins_height[1:])/2
centred_P = (bins_P[:-1] + bins_P[1:])/2

spline = scipy.interpolate.interp2d(centred_height, centred_P, hist, kind='cubic')
spline_func = lambda H, P: float(spline(float(H), float(P)))  # needed to avoid casting errors
spline_diff = lambda H, P: mpmath.diff(spline_func, (H, P), (0, 1), h=1e-6)

mpmath.dps = 100
integral = -mpmath.quadts(lambda H, P: spline_func(H, P)/H + (P/H)*spline_diff(H, P), (bins_height[0], bins_height[-1]), (bins_P[0] if bins_P[0] > 0 else 0, bins_P[-1]))

print(integral)
