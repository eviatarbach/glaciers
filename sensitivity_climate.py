import pickle

import numpy

from data import RGI_REGIONS, p, gamma, final_volume_vec, diff_vec

with open('data/serialized/all_glaciers', 'br') as all_glaciers_file:
    all_glaciers = pickle.load(all_glaciers_file).dropna()

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

        volumes = region['volume']
        heights = region['Thickness']
        lengths = region['LENGTH']
        slopes = region['SLOPE_avg']*numpy.pi/180
        areas = region['area']
        g_abl = region['g_abl']/1000
        g_acc = region['g_acc']/1000

        G = g_acc/g_abl - 1

        cl = volumes/(lengths**p)
        ca = volumes/(areas**gamma)

        Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))

        volumes_nd = volumes/Ldim**3
        cl_nd = cl*Ldim**(p - 3)
        ca_nd = ca*Ldim**(2*gamma - 3)

        # Mid-range
        ela_mid = (region['Zmax'] + region['Zmin'])/2

        # Area-weighted
        ela_weighted = region['ELA']

        # Hypsometric median
        ela_median = region['ELA2']

        # assert(sum(zela > heights) == 0)

        for ela_label, ela in (('mid', ela_mid), ('weighted', ela_weighted),
                               ('median', ela_median)):
            # Steady
            # zela = heights - lengths*slopes/2
            # zela_nd = zela/Ldim
            # P = (2*zela_nd*cl_nd**(1/q))/(slopes)

            zela = ela/1000 - (region['Zmax']/1000 - region['Thickness'])
            zela_nd = zela/Ldim
            P = zela_nd/(ca_nd**(1/gamma))

            all_glaciers.loc[(region_name,),
                             'P_{ela_label}'.format(ela_label=ela_label)] = P.values

            sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)

            all_glaciers.loc[(region_name,), ('sensitivity_{ela_label}'
                                              .format(ela_label=ela_label))] = sensitivity.values

            volumes_ss = final_volume_vec(G, P, volumes_nd)

            tau = -(1/20*G*P**2/volumes_ss**(4/5) - 1/5*G*P/volumes_ss**(3/5)
                    - 4/5*P/volumes_ss**(1/5) + 3/20*G/volumes_ss**(2/5) - 7/5*volumes_ss**(2/5)
                    + 1)**(-1)*g_abl**(-1)

            all_glaciers.loc[(region_name,),
                             'tau_{ela_label}'.format(ela_label=ela_label)] = tau.values

            # Also record results for G*=0, g_abl=g
            sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(0, P, volumes_nd)

            #region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])
            #region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

            #sensitivity = sensitivity[sensitivity > -5000]

            #sensitivities.append(sensitivity)
            #region_volumes.append(region_volume)
