import pickle

import numpy
import matplotlib.pyplot as plt
import mpmath

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

f = pickle.load(open('interpolant.p', 'rb'))

poly = numpy.poly1d(numpy.polyfit(f.x, f.y, deg=30))

AAR = 0.6

a = 0.6
q = 2.2
gamma = 1.375

mat = loadmat('data.mat')['Regions']

all_P = numpy.array([])
all_L = []
sensitivities = []

for i, region_name in enumerate(RGI_REGIONS):
    region = mat[region_name]

    volumes = region['volumes']
    heights = region['heights']
    lengths = region['lengths']
    slopes = region['slopes']
    areas = region['areas']

    zela = heights - AAR*lengths*numpy.tan(slopes)

    cl = volumes/(lengths**q)
    ca = volumes/(areas**gamma)
    cw = (ca/cl)**(1/(q - gamma))

    Ldim = ((2*cl**((a + 2)/q)*cw**a)/slopes)**(q/(3*(a - q + 2)))

    zela_nd = zela/Ldim
    cl_nd = cl*Ldim**(q - 3)

    P = (2*zela_nd*cl_nd**(1/q))/slopes

    indices = numpy.where((-2.53 > P) & (P > -2583))
    P = P[indices]
    Ldim = Ldim[indices]
    cl = cl[indices]
    slopes = slopes[indices]
    lengths = lengths[indices]

    all_L.append(lengths)

    all_P = numpy.hstack([all_P, P])

    plt.subplot(5, 4, i + 1)
    plt.title(RGI_NAMES[i])
    plt.hist(numpy.sign(P)*numpy.log(numpy.abs(P) + 1), 100, edgecolor='none', color='#C6D4E1')
    plt.xlim(xmin=-12, xmax=2)
    plt.axvline(0.1705)
    plt.xlabel('$\operatorname{sgn}(P^*)\log(|P^*| + 1)$')
    plt.ylabel('$N$', rotation=0)

    diff = numpy.vectorize(lambda p: mpmath.diff(lambda p2: poly(p2), p, h=1e-6))
    sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/slopes*diff(P)
    sensitivities.append(sensitivity)
    #sum((Ldim**3)*(poly((2*((zela + 0.00001)/Ldim)*cl_nd**(1/q))/slopes) - poly(P))/0.00001)

plt.show()
