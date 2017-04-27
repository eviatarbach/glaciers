"""
Perform Sobol sensitivity analysis on the climate sensitivity and response time of glaciers.

This is done by estimating the empirical distributions of the model parameters, sampling from the
distributions using low-discrepancy sequences, and computing the Sobol estimators. Note that Sobol
sensitivity analysis assumes the parameter distributions are independent, which is not true in this
case.
"""

import pandas
import numpy
import scipy.stats
from SALib.sample import saltelli
from SALib.analyze import sobol
from data import p, gamma, diff_vec, final_volume_vec
from scipy.interpolate import interp1d


# Modified version of function from statsmodels.distributions.empirical_distribution
def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function x (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """
    x = numpy.asarray(x)
    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = numpy.array(y)

    a = numpy.argsort(y)

    return interp1d(y[a], x[a], bounds_error=False, fill_value='extrapolate')


all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

G = all_glaciers['g_acc']/all_glaciers['g_abl'] - 1
cl = all_glaciers['volume']/all_glaciers['Lmax']**p
ca = all_glaciers['volume']/all_glaciers['area']**gamma

# Fit the empirical distributions
cl_params = scipy.stats.lognorm.fit(cl)
volume_params = scipy.stats.lognorm.fit(all_glaciers['volume'])

# TODO: remove repeated processing
all_glaciers['ELA_mid'] = (all_glaciers['Zmax'] + all_glaciers['Zmin'])/2
all_glaciers = all_glaciers.replace(-numpy.inf, numpy.nan)
ela = all_glaciers[['ELA_mid', 'ELA_weighted', 'ELA_median']].mean(axis=1)

zela = ela - (all_glaciers['Zmax'] - all_glaciers['THICK_mean'])

# Estimate empirical distribution with kernel density estimation, then invert the CDF for use in
# inverse transform sampling


def kde_ppf(data):
    """
    Return the inverse CDF (PPF) of the empirical distribution.

    The empirical distribution is approximated through Gaussian kernel density estimation of the
    given data.
    """
    kde = scipy.stats.gaussian_kde(data)

    def cdf(x):
        return kde.integrate_box_1d(min(data), x)

    return monotone_fn_inverter(numpy.vectorize(cdf), numpy.linspace(min(data), max(data), 1000))


zela_ppf = kde_ppf(zela)
G_ppf = kde_ppf(G)
ca_ppf = kde_ppf(ca)
slopes_ppf = kde_ppf(all_glaciers['SLOPE_avg'].dropna())
lapse_rate_ppf = kde_ppf(all_glaciers['lapse_rate'])
g_acc_ppf = kde_ppf(all_glaciers['g_acc'])
g_abl_ppf = kde_ppf(all_glaciers['g_abl'])

# Set unif(0, 1) distributions for inverse transform sampling
prob_sens = {'num_vars': 7, 'names': ['G', 'zela', 'ca', 'cl', 'slope', 'volume', 'lapse_rate'],
             'bounds': [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                        [0.0, 1.0]],
             'dists': ['unif', 'unif', 'unif', 'unif', 'unif', 'unif', 'unif']}
prob_tau = {'num_vars': 7, 'names': ['g_acc', 'g_abl', 'zela', 'ca', 'cl', 'slope', 'volume'],
            'bounds': [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                       [0.0, 1.0]],
            'dists': ['unif', 'unif', 'unif', 'unif', 'unif', 'unif', 'unif']}

# Generate parameter values
sens_sample = saltelli.sample(prob_sens, 10000, calc_second_order=True)
tau_sample = saltelli.sample(prob_tau, 10000, calc_second_order=True)


def sens_glacier(param_vals):
    """Compute the sensitivity of the glaciers with parameters given in `param_vals`."""
    G = G_ppf(param_vals[:, 0])
    zela = zela_ppf(param_vals[:, 1])
    ca = ca_ppf(param_vals[:, 2])
    slopes = slopes_ppf(param_vals[:, 4])
    lapse_rate = lapse_rate_ppf(param_vals[:, 6])
    cl = scipy.stats.lognorm.ppf(param_vals[:, 3], *cl_params)
    volumes = scipy.stats.lognorm.ppf(param_vals[:, 5], *volume_params)
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)*lapse_rate**(-1)
    return sensitivity


def tau_glacier(param_vals):
    """Compute the response time of the glaciers with parameters given in `param_vals`."""
    g_acc = g_acc_ppf(param_vals[:, 0])
    g_abl = g_abl_ppf(param_vals[:, 1])
    G = g_acc/g_abl - 1
    zela = zela_ppf(param_vals[:, 2])
    ca = ca_ppf(param_vals[:, 3])
    slopes = slopes_ppf(param_vals[:, 5])
    cl = scipy.stats.lognorm.ppf(param_vals[:, 4], *cl_params)
    volumes = scipy.stats.lognorm.ppf(param_vals[:, 6], *volume_params)
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    volumes_ss = final_volume_vec(G, P, volumes_nd)*Ldim**3
    tau = -(1/20*G*P**2/volumes_ss**(4/5) - 1/5*G*P/volumes_ss**(3/5)
            - 4/5*P/volumes_ss**(1/5) + 3/20*G/volumes_ss**(2/5) - 7/5*volumes_ss**(2/5)
            + 1)**(-1)*g_abl**(-1)
    tau[volumes_ss == 0] = 0
    return tau


# Calculate model output
sensitivity = sens_glacier(sens_sample)
tau = tau_glacier(tau_sample)

# Complete Sobol sensitivity analysis
sobol_sens = sobol.analyze(prob_sens, sensitivity, calc_second_order=True)
sobol_tau = sobol.analyze(prob_tau, tau, calc_second_order=True)
