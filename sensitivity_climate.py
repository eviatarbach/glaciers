import multiprocessing
import pickle

import pandas
import numpy
import scipy.stats

from data import RGI_REGIONS, p, gamma, final_volume_vec, diff_vec, ELA_CONV, ERRS
from bifurcation_distance import P0_vec


def sample_reject(means, cov, a, b):
    samples = numpy.zeros([len(means), 2])
    for i, mean in enumerate(means):
        dist_acc = scipy.stats.norm(loc=mean[0], scale=cov[0][0])
        dist_abl = scipy.stats.norm(loc=mean[1], scale=cov[1][1])
        sample = numpy.vstack([dist_acc.rvs(size=10), dist_abl.rvs(size=10)]).T
        mask = ((sample[:, 1] > 0) & (sample[:, 0]/sample[:, 1] < b)
                & (sample[:, 0]/sample[:, 1] > a))
        while sum(mask) == 0:
            sample = numpy.vstack([dist_acc.rvs(size=10), dist_abl.rvs(size=10)]).T
            mask = ((sample[:, 1] > 0) & (sample[:, 0]/sample[:, 1] < b)
                    & (sample[:, 0]/sample[:, 1] > a))
        samples[i, :] = sample[numpy.where(mask)[0][0]]
    return samples[:, 0], samples[:, 1]


all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')
glaciers = pandas.read_pickle('data/serialized/glaciers_climate')

# remove all glaciers whose ELA is above or below the glacier, for some reason
# all_glaciers = all_glaciers[(all_glaciers['Zmin'] < all_glaciers['ELA'])
#                             & (all_glaciers['ELA'] < all_glaciers['Zmax'])]

region_volumes = []

all_glaciers = all_glaciers.sort_index(level=[0, 1])


def run(_, ensemble=True):
    if ensemble:
        with lock:
            iteration.value += 1
            print(iteration.value)
    numpy.random.seed()
    run_data = all_glaciers.copy()
    for i, region_name in enumerate(RGI_REGIONS):
        region = run_data.loc[region_name]

        # Restore all heights that were less than 0
        # heights = heights + (heights <= 0)*region['THICK_mean']

        slopes = region['SLOPE_avg']
        if ensemble:
            scale = 0.255*slopes**3.349
            slopes = scipy.stats.truncnorm(a=-slopes/scale, b=numpy.inf, loc=slopes,
                                           scale=scale).rvs(size=len(slopes))

        areas = region['area']
        if ensemble:
            scale = 7.3822*areas**0.7
            areas = scipy.stats.truncnorm(a=-areas/scale, b=numpy.inf, loc=areas,
                                          scale=scale).rvs(size=len(areas))

        interp_volume_mask = region['interp_volume']

        volumes = region['volume']
        heights = region['THICK_mean']
        if ensemble:
            heights = scipy.stats.lognorm(scale=heights, s=ERRS['height']).rvs(size=len(heights))

        # Multiplying area by height instead of using volume directly since we have uncertainty
        # estimates provided in the height, not in the volume.
        volumes.loc[~interp_volume_mask] = (areas*heights)[~interp_volume_mask]

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

        g_acc = region['g_acc'].values
        g_abl = region['g_abl'].values

        if ensemble:
            g_acc, g_abl = sample_reject(means=numpy.vstack([g_acc, g_abl]).T,
                                         cov=numpy.diag([ERRS['g_acc'], ERRS['g_abl']]),
                                         a=(glaciers['g_acc']/glaciers['g_abl']).min(),
                                         b=(glaciers['g_acc']/glaciers['g_abl']).max())

        G = g_acc/g_abl - 1

        run_data.loc[(region_name,), 'G'] = G
        run_data.loc[(region_name,), 'g_abl'] = g_abl
        run_data.loc[(region_name,), 'g_acc'] = g_acc

        cl = volumes/(lengths**p)
        ca = volumes/(areas**gamma)

        Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))

        volumes_nd = volumes/Ldim**3

        run_data.loc[(region_name,), 'volumes_nd'] = volumes_nd.values

        cl_nd = cl*Ldim**(p - 3)
        ca_nd = ca*Ldim**(2*gamma - 3)

        ela = region['ELA_weighted']

        # prefer area-weighted, if missing use mid-range
        ela_mask = ela.isnull()
        ela[ela_mask] = region['ELA_mid'][ela_mask]
        ela_conv = ELA_CONV['ela_weighted']*numpy.ones(len(ela))
        ela_conv[ela_mask] = ELA_CONV['ela_mid']
        ela = ela + ela_conv
        if ensemble:
            ela_err = ERRS['ela_weighted']*numpy.ones(len(ela))
            ela_err[ela_mask] = ERRS['ela_mid']
            ela = scipy.stats.norm(loc=ela, scale=ela_err).rvs(size=len(ela))

        zela = ela - (region['Zmax'] - heights)
        zela_nd = zela/Ldim
        P = zela_nd/(ca_nd**(1/gamma))

        run_data.loc[(region_name,), 'zela'] = zela.values
        run_data.loc[(region_name,), 'P'] = P.values

        bif_dist = P0_vec(G) - P
        run_data.loc[(region_name,), 'bif_dist'] = (bif_dist*(zela/P)).values

        sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)

        run_data.loc[(region_name,), 'sensitivity'] = sensitivity.values

        volumes_ss = final_volume_vec(G, P, volumes_nd)

        run_data.loc[(region_name,), 'volumes_ss'] = (volumes_ss*Ldim**3).values

        tau = -(1/20*G*P**2/volumes_ss**(4/5) - 1/5*G*P/volumes_ss**(3/5)
                - 4/5*P/volumes_ss**(1/5) + 3/20*G/volumes_ss**(2/5) - 7/5*volumes_ss**(2/5)
                + 1)**(-1)*g_abl**(-1)

        run_data.loc[(region_name,), 'tau'] = tau.values

    return run_data[['P', 'G', 'sensitivity', 'volumes_ss', 'tau', 'bif_dist']]


def run_all(n_samples=100):
    global iteration
    global lock
    iteration = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    all_data = pool.map(run, range(n_samples))
    pickle.dump(all_data, open('data/serialized/all_data', 'wb'))


def run_single():
    single_data = run(0, ensemble=False)
    pandas.to_pickle(single_data, 'data/serialized/single_data')
