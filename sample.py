import pandas
import numpy

from data import p, gamma

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)

G = all_glaciers['G']
cl = all_glaciers['volume']/all_glaciers['length']**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma

dat_all = pandas.concat([G, all_glaciers['g_abl'], numpy.log(ca), numpy.log(cl),
                         numpy.arctan(all_glaciers['SLOPE_avg']),
                         numpy.log(all_glaciers['volume'])], axis=1)
dat_all.columns = ['G', 'g_abl', 'ca', 'cl', 'slope', 'volume']
dat_all = dat_all[~numpy.isnan(dat_all).any(axis=1)]

all_means = dat_all.mean(axis=0)
all_stds = dat_all.std(axis=0)

# Normalize
dat_all = (dat_all - all_means)/all_stds


def sens_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    param_vals = param_vals*all_stds + all_means
    G = param_vals['G']
    ca = numpy.exp(param_vals['ca'])
    cl = numpy.exp(param_vals['cl'])
    slopes = numpy.tan(param_vals['slope'])
    volumes = numpy.exp(param_vals['volume'])
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    V_0 = (G*(gamma - 1)/(2*(numpy.sqrt(G + 1) - 1)*(2 - gamma)))**(gamma/(3 - 2*gamma))
    volumes_nd = volumes/Ldim**3
    diff = gamma*G*volumes_nd**((2*gamma + 1)/gamma)/(2*(gamma - 2)*(numpy.sqrt(G + 1) - 1)*volumes_nd**(3/gamma) + (gamma - 1)*G*volumes_nd**2)
    sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff
    sensitivity[volumes_nd < V_0] = 0
    sensitivity[numpy.isnan(sensitivity)] = 0
    return sensitivity


def tau_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    param_vals = param_vals*all_stds + all_means
    G = param_vals['G']
    g_abl = param_vals['g_abl']
    ca = numpy.exp(param_vals['ca'])
    cl = numpy.exp(param_vals['cl'])
    slopes = numpy.tan(param_vals['slope'])
    volumes = numpy.exp(param_vals['volume'])
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    V_0 = (G*(gamma - 1)/(2*(numpy.sqrt(G + 1) - 1)*(2 - gamma)))**(gamma/(3 - 2*gamma))
    volumes_nd = volumes/Ldim**3
    P = (G*volumes_nd**2 - 2*volumes_nd**(3/gamma)*(numpy.sqrt(G + 1) - 1))/(G*volumes_nd**((1 + gamma)/gamma))
    tau = -(1/20*G*P**2/volumes_nd**(4/5) - 1/5*G*P/volumes_nd**(3/5)
            - 4/5*P/volumes_nd**(1/5) + 3/20*G/volumes_nd**(2/5) - 7/5*volumes_nd**(2/5)
            + 1)**(-1)*g_abl**(-1)
    tau[volumes_nd < V_0] = 0
    tau[numpy.isnan(tau)] = 0
    return tau


def bif_dist_glacier(param_vals):
    param_vals = param_vals*all_stds + all_means
    G = param_vals['G']
    ca = numpy.exp(param_vals['ca'])
    cl = numpy.exp(param_vals['cl'])
    slopes = numpy.tan(param_vals['slope'])
    volumes = numpy.exp(param_vals['volume'])
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    volumes_nd = volumes/Ldim**3
    P = (-2*volumes_nd**(3/gamma)*(numpy.sqrt(G + 1) - 1) + G*volumes_nd**2)/(G*volumes_nd**((1 + gamma)/gamma))
    P_0 = ((3 - 2*gamma)/(2 - gamma))*(G*(gamma - 1)/(2*(2 - gamma)*(numpy.sqrt(G + 1) - 1)))**((gamma - 1)/(3 - 2*gamma))
    conversion = (slopes**(gamma - 1)/(2**(gamma - 1)*ca**((2 - gamma)/gamma)*cl**((2 - gamma)*(gamma - 1)/gamma)))**(1/(2*gamma - 3))
    bif_dist = (P_0 - P)*conversion
    bif_dist[numpy.isnan(bif_dist)] = 0
    return bif_dist


def sample_all(n):
    samples = dat_all.iloc[numpy.random.choice(dat_all.shape[0], n)]
    return samples
