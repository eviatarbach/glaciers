import multiprocessing
import pickle

import pandas
import numpy
import scipy.stats

from data import RGI_REGIONS, p, gamma, final_volume_vec, diff_vec

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2

all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)

# remove all glaciers whose ELA is above or below the glacier, for some reason
# all_glaciers = all_glaciers[(all_glaciers['Zmin'] < all_glaciers['ELA'])
#                             & (all_glaciers['ELA'] < all_glaciers['Zmax'])]

region_volumes = []

all_glaciers = all_glaciers.sort_index(level=[0, 1])

# Drop glaciers that have no slope information
all_glaciers = all_glaciers[~all_glaciers['SLOPE_avg'].isnull() | ~all_glaciers['Slope'].isnull()]
# all_glaciers = all_glaciers[all_glaciers['Slope'] != 0]

iterations = 0

ERRS = {'slope': 0.029, 'height': 0.3, 'vol_interp': 0.223, 'length': 0.2, 'length_interp': 0.249,
        'g_abl': 0.004774, 'g_acc': 0.001792}


def run(i, ensemble=True):
    numpy.random.seed()
    global iterations
    iterations += 1
    print(iterations)
    run_data = all_glaciers.copy()
    for i, region_name in enumerate(RGI_REGIONS):
        region = run_data.loc[region_name]

        # Restore all heights that were less than 0
        # heights = heights + (heights <= 0)*region['THICK_mean']

        slopes = region['SLOPE_avg']
        if ensemble:
            slopes = scipy.stats.truncnorm(a=0, b=numpy.inf, loc=slopes,
                                           scale=ERRS['slope']).rvs(size=len(slopes))

        areas = region['area']
        if ensemble:
            areas = scipy.stats.truncnorm(a=0, b=numpy.inf, loc=areas,
                                          scale=7.3822*areas**0.7).rvs(size=len(areas))

        interp_volume_mask = region['interp_volume']

        volumes = region['volume']
        heights = region['THICK_mean']
        if ensemble:
            heights = scipy.stats.lognorm(scale=heights, s=ERRS['height']).rvs(size=len(heights))

        # Multiplying area by height instead of using volume directly since we have uncertainty
        # estimates provided in the height, not in the volume.
        volumes.loc[~interp_volume_mask] = areas*heights

        if ensemble:
            # For the rest of the volumes, we need to add the interpolation error
            volumes.loc[interp_volume_mask] = scipy.stats.lognorm(scale=volumes[interp_volume_mask],
                                                                  s=ERRS['vol_interp']).rvs(size=sum(interp_volume_mask))

        lengths = region['Lmax']
        interp_length_mask = region['interp_length']

        if ensemble:
            lengths.loc[~interp_length_mask] = scipy.stats.lognorm(scale=lengths[~interp_length_mask],
                                                                   s=ERRS['length']).rvs(size=sum(~interp_length_mask))

            lengths.loc[interp_length_mask] = scipy.stats.lognorm(scale=lengths[interp_length_mask],
                                                                  s=ERRS['length_interp']).rvs(size=sum(interp_length_mask))

        g_abl = region['g_abl']
        g_acc = region['g_acc']

        if ensemble:
            g_abl = scipy.stats.truncnorm(a=0, b=numpy.inf, loc=g_abl,
                                          scale=ERRS['g_abl']).rvs(size=len(g_abl))

            g_acc = scipy.stats.norm(loc=g_acc, scale=ERRS['g_acc']).rvs(size=len(g_acc))
        #g = truncated_normal(region['g'], 0.002712*numpy.ones(len(region['g'])))

        G = g_acc/g_abl - 1

        run_data.loc[(region_name,), 'G'] = G.values

        cl = volumes/(lengths**p)
        ca = volumes/(areas**gamma)

        Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))

        volumes_nd = volumes/Ldim**3
        cl_nd = cl*Ldim**(p - 3)
        ca_nd = ca*Ldim**(2*gamma - 3)

        # TODO: fix when only one is available, in which case error is 0
        ela = region[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean(axis=1)
        if ensemble:
            ela = scipy.stats.norm(loc=ela,
                                   scale=region[['ELA_mid', 'ELA_weighted',
                                                 'ELA_median']].std(axis=1).replace(numpy.nan, 0) + 1e-8).rvs(size=len(ela))

        # assert(sum(zela > heights) == 0)

        # Steady
        # zela = heights - lengths*slopes/2
        # zela_nd = zela/Ldim
        # P = (2*zela_nd*cl_nd**(1/q))/(slopes)

        zela = ela - (region['Zmax'] - heights)
        zela_nd = zela/Ldim
        P = zela_nd/(ca_nd**(1/gamma))

        run_data.loc[(region_name,), 'P'] = P.values

        sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)

        run_data.loc[(region_name,), 'sensitivity'] = sensitivity.values

        volumes_ss = final_volume_vec(G, P, volumes_nd)

        run_data.loc[(region_name,), 'volumes_ss'] = (volumes_ss*Ldim**3).values

        tau = -(1/20*G*P**2/volumes_ss**(4/5) - 1/5*G*P/volumes_ss**(3/5)
                - 4/5*P/volumes_ss**(1/5) + 3/20*G/volumes_ss**(2/5) - 7/5*volumes_ss**(2/5)
                + 1)**(-1)*g_abl**(-1)

        run_data.loc[(region_name,), 'tau'] = tau.values

        # Also record results for G*=0, g_abl=g
        # sensitivity_gz = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(0, P, volumes_nd)

        # run_data.loc[(region_name,), 'sensitivity_gz'] = sensitivity_gz

        # volumes_ss_gz = final_volume_vec(0, P, volumes_nd)

        # run_data.loc[(region_name,), 'volumes_ss_gz'] = volumes_ss_gz*Ldim**3

        # tau_gz = (4/5*P/volumes_ss_gz**(1/5) + 7/5*volumes_ss_gz**(2/5) - 1)**(-1)*g**(-1)

        # run_data.loc[(region_name,), 'tau_gz'] = tau_gz
    return run_data[['P', 'G', 'sensitivity', 'volumes_ss', 'tau']]


def run_all():
    pool = multiprocessing.Pool(processes=4)
    all_data = pool.map(run, range(100))
    pickle.dump(all_data, open('all_data', 'wb'))
# all_glaciers.to_pickle('data/serialized/all_glaciers')
        #region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])
        #region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

        #sensitivity = sensitivity[sensitivity > -5000]

        #sensitivities.append(sensitivity)
        #region_volumes.append(region_volume)
