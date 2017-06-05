import pandas
import numpy
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from data import p, gamma, diff_vec, final_volume_vec

SUBSET_SIZE = 10000

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

# Normalize
dat_sens = (dat_sens - sens_means)/sens_stds
dat_tau = (dat_tau - tau_means)/tau_stds

# Pick smaller subset, to reduce cost of evaluating the PDF of the kernel density estimation
numpy.random.shuffle(dat_sens)
dat_sens_subset = dat_sens[:SUBSET_SIZE, :]

numpy.random.shuffle(dat_tau)
dat_tau_subset = dat_tau[:SUBSET_SIZE, :]


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


def sample_joint_sens(n, subset=range(1, dat_sens.shape[1] + 1)):
    if isinstance(subset, int):
        subset = [subset]
    subset = [i - 1 for i in subset]
    samples = dat_sens[numpy.random.choice(dat_sens.shape[0], n), :][:, subset].tolist()
    return samples


def conditional_PDF_sens(Sj, Sjc, xjc):
    if isinstance(Sj, int):
        Sj = [Sj]
    if isinstance(Sjc, int):
        Sjc = [Sjc]
    if not isinstance(xjc, list):
        xjc = [xjc]
    Sj = [i - 1 for i in Sj]
    Sjc = [i - 1 for i in Sjc]
    cond = KDEMultivariateConditional(endog=dat_sens_subset[:, Sj], exog=dat_sens_subset[:, Sjc],
                                      bw='normal_reference', dep_type='c'*len(Sj),
                                      indep_type='c'*len(Sjc))

    def conditional(yj):
        if not isinstance(yj, list):
            yj = [yj]
        return cond.pdf(endog_predict=yj, exog_predict=xjc).tolist()

    return conditional


def sample_joint_tau(n, subset=range(1, dat_tau.shape[1] + 1)):
    if isinstance(subset, int):
        subset = [subset]
    subset = [i - 1 for i in subset]
    samples = dat_tau[numpy.random.choice(dat_tau.shape[0], n), :][:, subset].tolist()
    return samples


def conditional_PDF_tau(Sj, Sjc, xjc):
    if isinstance(Sj, int):
        Sj = [Sj]
    if isinstance(Sjc, int):
        Sjc = [Sjc]
    if not isinstance(xjc, list):
        xjc = [xjc]
    Sj = [i - 1 for i in Sj]
    Sjc = [i - 1 for i in Sjc]
    cond = KDEMultivariateConditional(endog=dat_tau_subset[:, Sj], exog=dat_tau_subset[:, Sjc],
                                      bw='normal_reference', dep_type='c'*len(Sj),
                                      indep_type='c'*len(Sjc))

    def conditional(yj):
        if not isinstance(yj, list):
            yj = [yj]
        return cond.pdf(endog_predict=yj, exog_predict=xjc).tolist()

    return conditional
