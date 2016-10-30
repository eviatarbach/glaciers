import multiprocessing

import pandas
import numpy

from data import RGI_REGIONS, p, gamma, final_volume_vec, diff_vec


def truncated_normal(vals, scale, truncate_at=0):
    sample = numpy.random.normal(0, scale)
    while any((vals + sample) <= truncate_at):
        nonpos = (vals + sample) <= truncate_at
        sample[nonpos.nonzero()[0]] = numpy.random.normal(0, scale[nonpos.nonzero()[0]])
    return vals + sample

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2

# remove all glaciers whose ELA is above or below the glacier, for some reason
# all_glaciers = all_glaciers[(all_glaciers['Zmin'] < all_glaciers['ELA'])
#                             & (all_glaciers['ELA'] < all_glaciers['Zmax'])]

region_volumes = []

all_glaciers = all_glaciers.sort_index(level=[0, 1])

# Drop glaciers that have no slope information
all_glaciers = all_glaciers[~all_glaciers['SLOPE_avg'].isnull() | ~all_glaciers['Slope'].isnull()]

iterations = 0

def run(i):
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

        # If no slope from Huss data, try RGI
        slopes.loc[slopes.isnull()] = region['Slope']
        slopes = truncated_normal(slopes, 0.029*numpy.ones(len(slopes)))

        areas = region['area']
        if region_name in ['Alaska', 'SouthernAndes']:
            areas = region['Area']
        areas.loc[areas.isnull()] = region['Area']
        areas = truncated_normal(areas, 0.213*areas)

        interp_volume_mask = region['interp_volume']

        volumes = region['volume']
        thickness = region['THICK_mean']
        # Also fill in heights
        missing_heights = thickness.isnull()
        thickness.loc[missing_heights] = volumes[missing_heights]/areas[missing_heights]
        heights = truncated_normal(thickness, 0.3*thickness)

        # Multiplying area by height instead of using volume directly since we have uncertainty
        # estimates provided in the height, not in the volume.
        volumes.loc[~interp_volume_mask] = areas*heights

        # For the rest of the volumes, we need to add the interpolation error
        volumes.loc[interp_volume_mask] = truncated_normal(volumes[interp_volume_mask],
                                                           0.223*volumes[interp_volume_mask])

        lengths = region['LENGTH']
        interp_length_mask = region['interp_length']

        lengths.loc[~interp_length_mask] = truncated_normal(lengths[~interp_length_mask],
                                                            1969*numpy.ones(sum(~interp_length_mask)))

        lengths.loc[interp_length_mask] = truncated_normal(lengths[interp_length_mask],
                                                           0.249*lengths[interp_length_mask])

        g_abl = truncated_normal(region['g_abl'], 0.004774*numpy.ones(len(region['g_abl'])))
        g_acc = truncated_normal(region['g_acc'], 0.001792*numpy.ones(len(region['g_acc'])),
                                 -numpy.inf)
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
        ela = truncated_normal(region[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean(axis=1),
                               region[['ELA_mid', 'ELA_weighted', 'ELA_median']].std(axis=1) + 1e-8)

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

pool = multiprocessing.Pool(processes=4)
all_data = pool.map(run, range(100))
# all_glaciers.to_pickle('data/serialized/all_glaciers')
        #region_volume = sum(volumes.values[((P != float('inf')) & (P < 0.1859) & (sensitivity > -5000)).nonzero()])
        #region_volume = sum(volumes.values[((P != float('inf')) & (P < P0)).nonzero()])

        #sensitivity = sensitivity[sensitivity > -5000]

        #sensitivities.append(sensitivity)
        #region_volumes.append(region_volume)
