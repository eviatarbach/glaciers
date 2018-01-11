import itertools

import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
import numpy
import pandas

import sample

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

sensitivity = importr('sensitivity')


def HSIC_sens(n_samples=500):
    X = sample.sample_all(n_samples)
    Y = sample.sens_glacier(X)
    mask = Y != 0
    res = sensitivity.sensiHSIC(X=X[mask], nboot=10)
    S = rpy2.robjects.r('tell')(res, y=numpy.log(-Y[mask]))[-1]
    rpy2.robjects.pandas2ri.ri2py(S).to_csv('data/HSIC_sens.txt')


def HSIC_tau(n_samples=500):
    X = sample.sample_all(n_samples)
    Y = sample.tau_glacier(X)
    mask = Y > 0
    res = sensitivity.sensiHSIC(X=X[mask], nboot=10)
    S = rpy2.robjects.r('tell')(res, y=numpy.log(Y[mask]))[-1]
    rpy2.robjects.pandas2ri.ri2py(S).to_csv('data/HSIC_tau.txt')


def HSIC_bif_dist(n_samples=500):
    X = sample.sample_all(n_samples)
    Y = sample.bif_dist_glacier(X)
    mask = Y > 0
    res = sensitivity.sensiHSIC(X=X[mask], nboot=10)
    S = rpy2.robjects.r('tell')(res, y=Y[mask])[-1]
    rpy2.robjects.pandas2ri.ri2py(S).to_csv('data/HSIC_bif_dist.txt')


def pairwise_HSIC(n_samples=1000):
    X = sample.sample_all(n_samples)
    labels = ['G', 'g_abl', 'ca', 'cl', 'slope', 'volume']
    pairs = itertools.combinations(labels, 2)
    df_HSIC = pandas.DataFrame(numpy.zeros([6, 6]), columns=labels, index=labels)
    for pair in pairs:
        res = sensitivity.sensiHSIC(X=X[[pair[0], pair[0]]].values)
        S = rpy2.robjects.r('tell')(res, y=X[[pair[1]]].values)[-1]
        df_HSIC[pair[0]].loc[pair[1]] = rpy2.robjects.pandas2ri.ri2py(S)['original'].iloc[0]
    return df_HSIC
