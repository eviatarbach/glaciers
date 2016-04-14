import pickle

import numpy
import scipy
import scipy.interpolate
import sklearn
import sklearn.linear_model
import mpmath
import statsmodels.api as sm

from data import RGI_REGIONS, RGI_NAMES, a, q, gamma, f_vec, alpha, delta, diff, P0

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

m = sm.GLM(glaciers['g'], sm.add_constant(Xnorm), family=sm.families.InverseGaussian(sm.families.links.log))

clf = sklearn.linear_model.RidgeCV()
clf.fit(Xnorm, glaciers['g'])

all_glaciers['g'] = m.fit().predict(sm.add_constant(Xnorm2))
#all_glaciers['g'] = clf.predict(Xnorm2)
#all_glaciers = all_glaciers[all_glaciers['g'] > 0]

region_volumes = []

sensitivities = []

all_glaciers = all_glaciers.sort_index(level=[0, 1])

for i, region_name in enumerate(RGI_REGIONS):
    if region_name in ['AntarcticSubantarctic', 'Alaska']:
        continue
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
        
        all_glaciers.loc[(region_name,), 'sensitivity'] = numpy.float64(sensitivity.values)

        volumes_ss = f_vec(P)

        tau = -(g*numpy.cos(slopes)*(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1)))**(-1)

        all_glaciers.loc[(region_name,), 'tau'] = tau.values

        #region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])
        region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

        #sensitivity = sensitivity[sensitivity > -5000]

        sensitivities.append(sensitivity)
        region_volumes.append(region_volume)
