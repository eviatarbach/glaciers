import math
import numpy
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

for region_name in RGI_REGIONS:
    region = mat[region_name]

    AAR = 0.6

    volumes = region['volumes']
    heights = region['heights']
    lengths = region['lengths']
    slopes = region['slopes']

    zela = heights - AAR*lengths*numpy.tan(slopes)
    P = 1 - zela/heights

    volume_sum = math.fsum(volumes)

    hist, bins, _ = plt.hist(P, bins=100, normed=True)
    centred = (bins[:-1] + bins[1:])/2
    spline = scipy.interpolate.interp1d(centred, hist, kind='cubic')
    x = numpy.linspace(centred[0], centred[-1], 1000)
    plt.plot(x, spline(x))

    threshold_P = 10

    last_val = centred[-1]
    for i, val in enumerate(centred):
        if val > threshold_P:
            last_index = i - 1
            last_val = centred[i - 1]
            break

    integral = scipy.integrate.romberg(lambda p: len(P)*p*spline(p), centred[0], last_val, divmax=100)
    s = sum(filter(lambda p: p > last_val, P))

    print((integral + s - sum(P))/sum(P))
