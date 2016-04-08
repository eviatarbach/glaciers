import pickle

import numpy
import scipy
import scipy.interpolate
import sklearn
import sklearn.linear_model
import mpmath

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

features = ['max_elevation', 'median_elevation', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover']
features2 = ['Zmax', 'Zmed', 'continentality', 'summer_temperature', 'precipitation', 'winter_precipitation', 'cloud_cover']

all_glaciers = pickle.load(open('all_glaciers.p', 'br')).dropna()
glaciers = pickle.load(open('glaciers', 'br')).dropna()

# Mass-balance gradient cannot be negative
glaciers = glaciers[glaciers['g'] > 0]

# remove all glaciers whose ELA is above or below the glacier, for some reason
all_glaciers = all_glaciers[(all_glaciers['Zmin'] < all_glaciers['ELA']) & (all_glaciers['ELA'] < all_glaciers['Zmax'])]

X = glaciers[features]
Xnorm = (X - X.mean())/(X.std())

X2 = all_glaciers.dropna()[features2]
Xnorm2 = (X2 - X2.mean())/(X2.std())

clf = sklearn.linear_model.RidgeCV()
clf.fit(Xnorm, glaciers['g'])

all_glaciers['g'] = clf.predict(Xnorm2)
all_glaciers = all_glaciers[all_glaciers['g'] > 0]

region_volumes = []

sensitivities = []

a = 0.6
q = 2.2
gamma = 1.375

P = pickle.load(open('P.p', 'rb'))
V = pickle.load(open('V.p', 'rb'))
P2 = pickle.load(open('P2.p', 'rb'))
V2 = pickle.load(open('V2.p', 'rb'))

f1 = scipy.interpolate.interp1d(P, V)
f2 = scipy.interpolate.interp1d(P2, V2)

alpha = 8/11
delta = 13/11

def F(P, V): return -V**(alpha)*P - V**(delta) + V

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

diff = numpy.vectorize(lambda p: mpmath.diff(lambda p2: f_vec(p2), p, h=1e-6))

for i, region_name in enumerate(RGI_REGIONS):
    print(region_name)
    region = all_glaciers.loc[region_name]

    if len(region):
        zela = region['ELA']/1000 - (region['Zmax']/1000 - region['Thickness'])

        volumes = region['volume']
        heights = region['Thickness']
        lengths = region['LENGTH']
        slopes = region['SLOPE_avg']*numpy.pi/180
        areas = region['area']
        g = region['g']

        assert(sum(zela > heights) == 0)

        cl = volumes/(lengths**q)
        ca = volumes/(areas**gamma)
        cw = (ca/cl)**(1/(q - gamma))

        Ldim = ((2*cl**((a + 2)/q)*cw**a)/(slopes*numpy.cos(slopes)))**(q/(3*(a - q + 2)))

        volumes_nd = volumes/Ldim**3

        zela_nd = zela/Ldim
        cl_nd = cl*Ldim**(q - 3)

        P = (2*zela_nd*cl_nd**(1/q))/(slopes*numpy.cos(slopes))

        all_glaciers.loc[(region_name,), 'P'] = P.values
        all_glaciers.loc[(region_name,), 'zela'] = zela.values

        sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/(slopes*numpy.cos(slopes))*diff(P)
        
        all_glaciers.loc[(region_name,), 'sensitivity'] = sensitivity.values

        tau = -g*(1 - alpha*P*(volumes_nd)**(alpha - 1) - delta*(volumes_nd)**(delta - 1))**(-1)

        all_glaciers.loc[(region_name,), 'tau'] = tau.values

        region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])

        sensitivity = sensitivity[sensitivity > -5000]

        sensitivities.append(sensitivity)
        region_volumes.append(region_volume)
