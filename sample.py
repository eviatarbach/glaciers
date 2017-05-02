import sklearn
import sklearn.neighbors
import pandas
import numpy
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from data import p, gamma, diff_vec

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2
all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)
ela = all_glaciers[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean(axis=1)

zela = ela - (all_glaciers['Zmax'] - all_glaciers['THICK_mean'])
G = all_glaciers['g_acc']/all_glaciers['g_abl'] - 1
cl = all_glaciers['volume']/all_glaciers['Lmax']**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma

dat = numpy.vstack([G, zela, numpy.log(ca), numpy.log(cl), all_glaciers['SLOPE_avg'],
                    numpy.log(all_glaciers['volume']), all_glaciers['lapse_rate']]).T
dat = dat[~numpy.isnan(dat).any(axis=1)]

dat_means = dat.mean(axis=0)
dat_stds = dat.std(axis=0)

# Normalize
dat = (dat - dat_means)/dat_stds

# Pick smaller subset, to reduce cost of evaluating the PDF of the kernel density estimation
numpy.random.shuffle(dat)
dat = dat[:50000, :]
# def joint(n):
#     k = sklearn.neighbors.KernelDensity(bandwidth=0.01)
#     k.fit(dat)
#     return k.sample(n).tolist()


# def conditional_PDF(Sj, Sjc, xjc):
#     # Switch to zero-based indexing
#     Sj = [i - 1 for i in Sj]
#     Sjc = [i - 1 for i in Sjc]
#     joint = sklearn.neighbors.KernelDensity(bandwidth=0.01).fit(dat)
#     marginal = sklearn.neighbors.KernelDensity(bandwidth=0.01).fit(dat[Sjc])
#
#     def conditional(yj):
#         pts = numpy.zeros([yj.shape[0], len(Sj) + len(Sjc)])
#         pts[:, Sj] = yj
#         pts[:, Sjc] = xjc
#         return joint.score_samples(pts) - marginal.score_samples(xjc)
#
#     return conditional


def sens_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    param_vals = pandas.DataFrame(param_vals).values
    param_vals = param_vals*dat_stds + dat_means
    G = param_vals[:, 0]
    zela = param_vals[:, 1]
    ca = numpy.exp(param_vals[:, 2])
    slopes = param_vals[:, 4]
    lapse_rate = param_vals[:, 6]
    cl = numpy.exp(param_vals[:, 3])
    volumes = numpy.exp(param_vals[:, 5])
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)*lapse_rate**(-1)
    return sensitivity.tolist()

# joint = KDEMultivariate(dat, bw='normal_reference', var_type='c'*dat.shape[1]).pdf
# def joint_PDF(yj):
#     return joint(yj).tolist()


def sample_joint(n, subset=range(1, dat.shape[1] + 1)):
    if isinstance(subset, int):
        subset = [subset]
    subset = [i - 1 for i in subset]
    samples = dat[numpy.random.choice(dat.shape[0], n), :][:, subset].tolist()
    return samples


def conditional_PDF(Sj, Sjc, xjc):
    # Switch to zero-based indexing
    # if Sj is None:
    #     Sj = numpy.empty([1, 0])
    # if Sjc is None:
    #     Sjc = numpy.empty([1, 0])
    # if xjc is None:
    #     xjc = numpy.empty([1, 0])
    if isinstance(Sj, int):
        Sj = [Sj]
    if isinstance(Sjc, int):
        Sjc = [Sjc]
    if not isinstance(xjc, list):
        xjc = [xjc]
    Sj = [i - 1 for i in Sj]
    Sjc = [i - 1 for i in Sjc]
    cond = KDEMultivariateConditional(endog=dat[:, Sj], exog=dat[:, Sjc], bw='normal_reference',
                                      dep_type='c'*len(Sj), indep_type='c'*len(Sjc))

    def conditional(yj):
        if not isinstance(yj, list):
            yj = [yj]
        return cond.pdf(endog_predict=yj, exog_predict=xjc).tolist()

    return conditional
