import pandas
import numpy

from data import p, gamma, diff_vec, final_volume_vec

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2
all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)
ela = all_glaciers[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean(axis=1)

zela = ela - (all_glaciers['Zmax'] - all_glaciers['THICK_mean'])
G = all_glaciers['g_acc']/all_glaciers['g_abl'] - 1
cl = all_glaciers['volume']/all_glaciers['Lmax']**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma

dat_sens = numpy.vstack([G, zela, numpy.log(ca), numpy.log(cl), all_glaciers['SLOPE_avg'],
                         numpy.log(all_glaciers['volume']), all_glaciers['lapse_rate']]).T
dat_sens = dat_sens[~numpy.isnan(dat_sens).any(axis=1)]

sens_means = dat_sens.mean(axis=0)
sens_stds = dat_sens.std(axis=0)

dat_tau = numpy.vstack([all_glaciers['g_acc'], all_glaciers['g_abl'], zela, numpy.log(ca),
                        numpy.log(cl), all_glaciers['SLOPE_avg'],
                        numpy.log(all_glaciers['volume'])]).T
dat_tau = dat_tau[~numpy.isnan(dat_tau).any(axis=1)]

tau_means = dat_tau.mean(axis=0)
tau_stds = dat_tau.std(axis=0)

dat_all = numpy.vstack([G, all_glaciers['g_acc'], all_glaciers['g_abl'], zela, numpy.log(ca),
                        numpy.log(cl), all_glaciers['SLOPE_avg'],
                        numpy.log(all_glaciers['volume']), all_glaciers['lapse_rate']]).T
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
    zela = param_vals[:, 1]
    ca = numpy.exp(param_vals[:, 2])
    cl = numpy.exp(param_vals[:, 3])
    slopes = param_vals[:, 4]
    volumes = numpy.exp(param_vals[:, 5])
    lapse_rate = param_vals[:, 6]
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)*lapse_rate**(-1)
    return sensitivity.tolist()


def tau_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    param_vals = pandas.DataFrame(param_vals).values
    param_vals = param_vals*tau_stds + tau_means
    g_acc = param_vals[:, 0]
    g_abl = param_vals[:, 1]
    zela = param_vals[:, 2]
    ca = numpy.exp(param_vals[:, 3])
    cl = numpy.exp(param_vals[:, 4])
    slopes = param_vals[:, 5]
    volumes = numpy.exp(param_vals[:, 6])
    G = g_acc/g_abl - 1
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    volumes_ss = final_volume_vec(G, P, volumes_nd)
    tau = -(1/20*G*P**2/volumes_ss**(4/5) - 1/5*G*P/volumes_ss**(3/5)
            - 4/5*P/volumes_ss**(1/5) + 3/20*G/volumes_ss**(2/5) - 7/5*volumes_ss**(2/5)
            + 1)**(-1)*g_abl**(-1)
    tau[volumes_ss == 0] = 0
    return tau.tolist()


def sample_joint_sens(n):
    samples = dat_sens[numpy.random.choice(dat_sens.shape[0], n), :].tolist()
    return samples


def sample_joint_tau(n):
    samples = dat_tau[numpy.random.choice(dat_tau.shape[0], n), :].tolist()
    return samples


def sample_joint_all(n):
    samples = dat_all[numpy.random.choice(dat_all.shape[0], n), :].tolist()
    return samples
