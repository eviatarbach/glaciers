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

hist_H, bins_H, _ = plt.hist(heights, bins=100, normed=True)
centred_H = (bins_H[:-1] + bins_H[1:])/2

spline_H = scipy.interpolate.interp1d(centred_H, hist_H, kind='cubic')

hist_z, bins_z, _ = plt.hist(zela, bins=100, normed=True)
centred_z = (bins_z[:-1] + bins_z[1:])/2

spline_z = scipy.interpolate.interp1d(centred_z, hist_z, kind='cubic')

integral1 = mpmath.quad(lambda H: float(spline_H(float(H)))/H, (centred_H[0], centred_H[-1]))

spline_diff = lambda z: mpmath.diff(lambda z: float(spline_z(float(z))), z, h=1e-6)

integral2 = mpmath.quad(lambda z: spline_diff(z), (centred_z[0] + 1e-6, centred_z[-1] - 1e-6))

integral3 = mpmath.quad(lambda z: spline_diff(z)*z, (centred_z[0] + 1e-6, centred_z[-1] - 1e-6))

integral = -integral1 + integral2 - integral1*integral3
