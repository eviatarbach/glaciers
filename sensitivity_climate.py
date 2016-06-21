import pickle

import numpy
import scipy
import scipy.interpolate
import sklearn
import sklearn.linear_model
import mpmath
import statsmodels.api as sm

from data import RGI_REGIONS, p, gamma

with open('data/serialized/glaciers_climate', 'br') as glaciers_file,\
     open('data/serialized/all_glaciers', 'br') as all_glaciers_file:
    all_glaciers = pickle.load(all_glaciers_file).dropna()
    glaciers = pickle.load(glaciers_file).dropna()

    # remove all glaciers whose ELA is above or below the glacier, for some reason
    all_glaciers = all_glaciers[(all_glaciers['Zmin'] < all_glaciers['ELA'])
                                & (all_glaciers['ELA'] < all_glaciers['Zmax'])]

    region_volumes = []

    sensitivities = []

    all_glaciers = all_glaciers.sort_index(level=[0, 1])

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
            g_abl = region['g_abl']
            g_acc = region['g_acc']

            #assert(sum(zela > heights) == 0)

            cl = volumes/(lengths**q)
            ca = volumes/(areas**gamma)

            Ldim = ((2*cl**((a + 2)/q)*cw**a)/(slopes))**(q/(3*(a - q + 2)))

            volumes_nd = volumes/Ldim**3
            cl_nd = cl*Ldim**(q - 3)

            # Steady
            zela = heights - lengths*slopes/2
            zela_nd = zela/Ldim
            P = (2*zela_nd*cl_nd**(1/q))/(slopes)

            all_glaciers.loc[(region_name,), 'P_steady'] = P.values

            sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/(slopes)*diff(P)

            all_glaciers.loc[(region_name,), 'sensitivity_steady'] = numpy.float64(sensitivity.values)

            volumes_ss = f_vec(P)

            tau = -(g*(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1)))**(-1)

            all_glaciers.loc[(region_name,), 'tau_steady'] = tau.values

            tau2 = -(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1))**(-1)
            all_glaciers.loc[(region_name,), 'tau2_steady'] = tau2.values

            # Mid-range
            ela = (region['Zmax'] + region['Zmin'])/2
            zela = ela/1000 - (region['Zmax']/1000 - region['Thickness'])
            zela_nd = zela/Ldim
            P = (2*zela_nd*cl_nd**(1/q))/(slopes)

            all_glaciers.loc[(region_name,), 'P_mid'] = P.values

            sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/(slopes)*diff(P)

            all_glaciers.loc[(region_name,), 'sensitivity_mid'] = numpy.float64(sensitivity.values)

            volumes_ss = f_vec(P)

            tau = -(g*(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1)))**(-1)

            all_glaciers.loc[(region_name,), 'tau_mid'] = tau.values

            tau2 = -(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1))**(-1)
            all_glaciers.loc[(region_name,), 'tau2_mid'] = tau2.values

            # Area-weighted
            ela = region['ELA']
            zela = ela/1000 - (region['Zmax']/1000 - region['Thickness'])
            zela_nd = zela/Ldim
            P = (2*zela_nd*cl_nd**(1/q))/(slopes)

            all_glaciers.loc[(region_name,), 'P_weighted'] = P.values

            sensitivity = Ldim**(3 - 3/q)*2*cl**(1/q)/(slopes)*diff(P)

            all_glaciers.loc[(region_name,), 'sensitivity_weighted'] = numpy.float64(sensitivity.values)

            volumes_ss = f_vec(P)

            tau = -(g*(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1)))**(-1)

            all_glaciers.loc[(region_name,), 'tau_weighted'] = tau.values

            tau2 = -(1 - alpha*P*(volumes_ss)**(alpha - 1) - delta*(volumes_ss)**(delta - 1))**(-1)
            all_glaciers.loc[(region_name,), 'tau2_weighted'] = tau2.values


            #all_glaciers.loc[(region_name,), 'zela'] = zela.values


            #region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])
            #region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

            #sensitivity = sensitivity[sensitivity > -5000]

            #sensitivities.append(sensitivity)
            #region_volumes.append(region_volume)
