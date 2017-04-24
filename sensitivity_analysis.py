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
    kde = scipy.stats.gaussian_kde(data)

    def cdf(x):
        return kde.integrate_box_1d(min(data), x)

    return monotone_fn_inverter(numpy.vectorize(cdf), numpy.linspace(min(data), max(data), 1000))


zela_ppf = kde_ppf(zela)
G_ppf = kde_ppf(G)
ca_ppf = kde_ppf(ca)
slopes_ppf = kde_ppf(all_glaciers['SLOPE_avg'].dropna())

# Set unif(0, 1) distributions for inverse transform sampling
prob = {'num_vars': 6, 'names': ['G', 'zela', 'ca', 'cl', 'slope', 'volume'],
        'bounds': [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        'dists': ['unif', 'unif', 'unif', 'unif', 'unif', 'unif']}

# Generate parameter values
param_vals = saltelli.sample(prob, 10000, calc_second_order=False)


def sens_glacier(param_vals):
    G = G_ppf(param_vals[:, 0])
    zela = zela_ppf(param_vals[:, 1])
    ca = ca_ppf(param_vals[:, 2])
    slopes = slopes_ppf(param_vals[:, 4])
    cl = scipy.stats.lognorm.ppf(param_vals[:, 3], *cl_params)
    volumes = scipy.stats.lognorm.ppf(param_vals[:, 5], *volume_params)
    Ldim = (2*ca**(1/gamma)*cl**(1/p)/slopes)**(gamma*p/(3*(gamma + p - gamma*p)))
    ca_nd = ca*Ldim**(2*gamma - 3)
    zela_nd = zela/Ldim
    volumes_nd = volumes/Ldim**3
    P = zela_nd/(ca_nd**(1/gamma))
    sensitivity = Ldim**(3/gamma)/ca**(1/gamma)*diff_vec(G, P, volumes_nd)
    volumes_ss = final_volume_vec(G, P, volumes_nd)*Ldim**3
    return sensitivity/volumes_ss


# Calculate model output
Y_dists_code = sens_glacier(param_vals)

# complete Sobol' sensitivity analysis
Si_dists_code = sobol.analyze(prob, Y_dists_code, calc_second_order=False)
