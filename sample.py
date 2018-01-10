import pandas
import numpy

from data import p, gamma

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)

lengths = (all_glaciers['Zmax'] - all_glaciers['Zmin'] - all_glaciers['THICK_mean'])/all_glaciers['SLOPE_avg']
all_glaciers = all_glaciers[lengths > 0]
lengths = lengths[lengths > 0]

G = all_glaciers['G']
cl = all_glaciers['volume']/lengths**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma

dat_sens = numpy.vstack([G, numpy.log(ca), numpy.log(cl), numpy.arctan(all_glaciers['SLOPE_avg']),
                         numpy.log(all_glaciers['volume'])]).T
dat_sens = dat_sens[~numpy.isnan(dat_sens).any(axis=1)]

sens_means = dat_sens.mean(axis=0)
sens_stds = dat_sens.std(axis=0)

dat_tau = numpy.vstack([G, all_glaciers['g_abl'], numpy.log(ca), numpy.log(cl),
                        numpy.arctan(all_glaciers['SLOPE_avg']),
                        numpy.log(all_glaciers['volume'])]).T
dat_tau = dat_tau[~numpy.isnan(dat_tau).any(axis=1)]

tau_means = dat_tau.mean(axis=0)
tau_stds = dat_tau.std(axis=0)

dat_all = numpy.vstack([G, all_glaciers['g_abl'], numpy.log(ca), numpy.log(cl),
                        numpy.arctan(all_glaciers['SLOPE_avg']),
                        numpy.log(all_glaciers['volume'])]).T
dat_all = dat_all[~numpy.isnan(dat_all).any(axis=1)]

all_means = dat_all.mean(axis=0)
all_stds = dat_all.std(axis=0)

# Normalize
dat_sens = (dat_sens - sens_means)/sens_stds
dat_tau = (dat_tau - tau_means)/tau_stds
dat_all = (dat_all - all_means)/all_stds


def sens_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    param_vals = pandas.DataFrame(param_vals).values
    param_vals = param_vals*sens_stds + sens_means
    G = param_vals[:, 0]
    ca = numpy.exp(param_vals[:, 1])
    cl = numpy.exp(param_vals[:, 2])
    slopes = numpy.tan(param_vals[:, 3])
    volumes = numpy.exp(param_vals[:, 4])
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
    param_vals = pandas.DataFrame(param_vals).values
    param_vals = param_vals*tau_stds + tau_means
    G = param_vals[:, 0]
    g_abl = param_vals[:, 1]
    ca = numpy.exp(param_vals[:, 2])
    cl = numpy.exp(param_vals[:, 3])
    slopes = numpy.tan(param_vals[:, 4])
    volumes = numpy.exp(param_vals[:, 5])
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
    param_vals = pandas.DataFrame(param_vals).values
    param_vals = param_vals*sens_stds + sens_means
    G = param_vals[:, 0]
    ca = numpy.exp(param_vals[:, 1])
    cl = numpy.exp(param_vals[:, 2])
    slopes = numpy.tan(param_vals[:, 3])
    volumes = numpy.exp(param_vals[:, 4])
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    volumes_nd = volumes/Ldim**3
    P = (-2*volumes_nd**(3/gamma)*(numpy.sqrt(G + 1) - 1) + G*volumes_nd**2)/(G*volumes_nd**((1 + gamma)/gamma))
    P_0 = ((3 - 2*gamma)/(2 - gamma))*(G*(gamma - 1)/(2*(2 - gamma)*(numpy.sqrt(G + 1) - 1)))**((gamma - 1)/(3 - 2*gamma))
    conversion = (slopes**(gamma - 1)/(2**(gamma - 1)*ca**((2 - gamma)/gamma)*cl**((2 - gamma)*(gamma - 1)/gamma)))**(1/(2*gamma - 3))
    bif_dist = (P_0 - P)*conversion
    bif_dist[numpy.isnan(bif_dist)] = 0
    return bif_dist


def sample_joint_sens(n):
    samples = dat_sens[numpy.random.choice(dat_sens.shape[0], n), :]
    return samples


def sample_joint_tau(n):
    samples = dat_tau[numpy.random.choice(dat_tau.shape[0], n), :]
    return samples


def sample_joint_bif_dist(n):
    samples = dat_sens[numpy.random.choice(dat_sens.shape[0], n), :]
    return samples


def sample_joint_all(n):
    samples = dat_all[numpy.random.choice(dat_all.shape[0], n), :]
    return samples
