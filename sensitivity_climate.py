from data import RGI_REGIONS, p, gamma, ERRS

import pandas
import numpy
from uncertainties import unumpy
sqrt = unumpy.sqrt
arctan = unumpy.arctan
tan = unumpy.tan


all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')
glaciers = pandas.read_pickle('data/serialized/glaciers_climate')

all_glaciers = all_glaciers.sort_index(level=[0, 1])


def run():
    run_data = all_glaciers.copy()
    for i, region_name in enumerate(RGI_REGIONS):
        print(region_name)
        region = run_data.loc[region_name]

        slopes = region['SLOPE_avg']
        slopes = unumpy.uarray(slopes, tan(0.255*arctan(slopes)**3.349))

        areas = region['area']
        areas = unumpy.uarray(areas, 7.3822*areas**0.7)

        interp_volume_mask = region['interp_volume']

        heights = unumpy.uarray(region['THICK_mean'], ERRS['height']*region['THICK_mean'])

        # Multiplying area by height instead of using volume directly since we have uncertainty
        # estimates provided in the height, not in the volume.
        volumes = areas*heights
        volumes[interp_volume_mask] = unumpy.uarray(region['volume'][interp_volume_mask],
                                                    ERRS['vol_interp']*region['volume'][interp_volume_mask])

        lengths = (region['Zmax'] - region['Zmin']
                   - region['THICK_mean'])/region['SLOPE_avg']
        lengths = unumpy.uarray(lengths, ERRS['length']*lengths)

        g_abl = unumpy.uarray(region['g_abl'].values, ERRS['g_abl'])
        G = unumpy.uarray(region['G'].values, ERRS['G'])

        mask_G = (unumpy.nominal_values(g_abl) > 0) & (unumpy.nominal_values(G) > -1)

        V_0 = (unumpy.nominal_values(G)*(gamma - 1)/(2*(numpy.sqrt(unumpy.nominal_values(G) + 1)
               - 1)*(2 - gamma)))**(gamma/(3 - 2*gamma))

        cl = volumes/(lengths**p)
        ca = volumes/(areas**gamma)

        Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
        run_data.loc[(region_name,), 'L0'] = unumpy.nominal_values(Ldim)

        volumes_nd = volumes/Ldim**3

        mask = mask_G & (unumpy.nominal_values(volumes_nd) > V_0) & ~unumpy.isnan(lengths)

        run_data.loc[(region_name,), 'G'] = unumpy.nominal_values(G)

        P = unumpy.uarray(numpy.zeros(len(region)), numpy.zeros(len(region)))
        P[mask] = (-2*volumes_nd[mask]**(3/gamma)*(sqrt(G[mask] + 1) - 1)
                   + G[mask]*volumes_nd[mask]**2)/(G[mask]*volumes_nd[mask]**((1 + gamma)/gamma))
        P[~mask] = numpy.nan
        run_data.loc[(region_name,), 'P'] = unumpy.nominal_values(P)

        P_0 = ((3 - 2*gamma)/(2 - gamma))*(unumpy.nominal_values(G[mask])*(gamma - 1)/(2*(2 - gamma)*(sqrt(unumpy.nominal_values(G[mask]) + 1) - 1)))**((gamma - 1)/(3 - 2*gamma))

        bif_dist = unumpy.uarray(numpy.zeros(len(region)), numpy.zeros(len(region)))
        conversion = (slopes[mask]**(gamma - 1)/(2**(gamma - 1)*ca[mask]**((2 - gamma)/gamma)*cl[mask]**((2 - gamma)*(gamma - 1)/gamma)))**(1/(2*gamma - 3))
        bif_dist[mask] = (P_0 - P[mask])*conversion
        bif_dist[~mask] = numpy.nan
        run_data.loc[(region_name,), 'bif_dist'] = unumpy.nominal_values(bif_dist)

        diff = gamma*G[mask]*volumes_nd[mask]**((2*gamma + 1)/gamma)/(2*(gamma - 2)*(sqrt(G[mask] + 1) - 1)*volumes_nd[mask]**(3/gamma) + (gamma - 1)*G[mask]*volumes_nd[mask]**2)

        sensitivity = unumpy.uarray(numpy.zeros(len(region)), numpy.zeros(len(region)))
        sensitivity[mask] = Ldim[mask]**3/conversion*diff
        sensitivity[~mask] = numpy.nan

        volumes[~mask] = 0
        volumes_nd[~mask] = 0

        run_data.loc[(region_name,), 'sensitivity'] = unumpy.nominal_values(sensitivity)
        run_data.loc[(region_name,), 'sensitivity_std'] = unumpy.std_devs(sensitivity)

        volumes_ss = volumes

        run_data.loc[(region_name,), 'volumes_ss'] = unumpy.nominal_values(volumes_ss)
        run_data.loc[(region_name,), 'volumes_ss_std'] = unumpy.std_devs(volumes_ss)

        tau = unumpy.uarray(numpy.zeros(len(region)), numpy.zeros(len(region)))
        tau[mask] = -(1/20*G[mask]*P[mask]**2/volumes_nd[mask]**(4/5)
                      - 1/5*G[mask]*P[mask]/volumes_nd[mask]**(3/5)
                      - 4/5*P[mask]/volumes_nd[mask]**(1/5)
                      + 3/20*G[mask]/volumes_nd[mask]**(2/5) - 7/5*volumes_nd[mask]**(2/5)
                      + 1)**(-1)*g_abl[mask]**(-1)
        tau[~mask] = numpy.nan

        run_data.loc[(region_name,), 'tau'] = unumpy.nominal_values(tau)
        run_data.loc[(region_name,), 'tau_std'] = unumpy.std_devs(tau)

    return run_data[['P', 'G', 'sensitivity', 'sensitivity_std', 'tau', 'tau_std', 'volumes_ss',
                     'volumes_ss_std', 'bif_dist', 'L0']]


def run_single():
    single_data = run()
    pandas.to_pickle(single_data, 'data/serialized/single_data')
