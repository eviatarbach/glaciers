import pandas
import numpy
from scipy.integrate import dblquad

import plot_config

import matplotlib.pyplot as plt

from data import ERRS, ICE_DENSITY

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')

g_abl_pts = numpy.linspace(glaciers['g_abl'].min()*ICE_DENSITY,
                           glaciers['g_abl'].max()*ICE_DENSITY, 25)
g_acc_pts = numpy.linspace(glaciers['g_acc'].min()*ICE_DENSITY,
                           glaciers['g_acc'].max()*ICE_DENSITY, 25)

a = (glaciers['g_acc']/glaciers['g_abl']).min()
b = (glaciers['g_acc']/glaciers['g_abl']).max()

mu_x = numpy.mean(glaciers['g_acc']*ICE_DENSITY)
mu_y = numpy.mean(glaciers['g_abl']*ICE_DENSITY)


def f(mu_x, mu_y, x, y):
    std_x, std_y = ERRS['g_acc']*ICE_DENSITY, ERRS['g_abl']*ICE_DENSITY
    return 1/(2*numpy.pi*std_x*std_y)*numpy.exp(-1/2*((x - mu_x)/std_x)**2
                                                - 1/2*((y - mu_y)/std_y)**2)


X, Y = numpy.meshgrid(g_acc_pts, g_abl_pts)


@numpy.vectorize
def std_acc(g_acc, g_abl):
    scaling = dblquad(lambda x, y: f(g_acc, g_abl, x, y), 0, numpy.inf,
                      lambda y: a*y, lambda y: b*y)[0]
    return numpy.sqrt(dblquad(lambda x, y: (x - g_acc)**2*f(g_acc, g_abl, x, y)/scaling, 0,
                              numpy.inf, lambda y: a*y, lambda y: b*y)[0])


@numpy.vectorize
def std_abl(g_acc, g_abl):
    scaling = dblquad(lambda x, y: f(g_acc, g_abl, x, y), 0, numpy.inf,
                      lambda y: a*y, lambda y: b*y)[0]
    return numpy.sqrt(dblquad(lambda x, y: (y - g_abl)**2*f(g_acc, g_abl, x, y)/scaling, 0,
                              numpy.inf, lambda y: a*y, lambda y: b*y)[0])


for grad in ['acc', 'abl']:
    if grad == 'acc':
        Z = std_acc(X, Y)
    else:
        Z = std_abl(X, Y)
    cp = plt.contourf(X, Y, Z, cmap='Blues_r')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.xlabel(r'$\dot{g}_\text{acc}$ (mm w.e./m/y)', fontsize=26)
    plt.ylabel(r'$\dot{g}_\text{abl}$ (mm w.e./m/y)', fontsize=26, rotation=0, labelpad=110)
    if grad == 'acc':
        plt.title(r'$\sqrt{\text{E}\left[(\dot{g}_\text{acc}'
                  r'- \mu_{\dot{g}_\text{acc}})^2\right]}$',
                  fontsize=26, y=1.05)
    else:
        plt.title(r'$\sqrt{\text{E}\left[(\dot{g}_\text{abl}'
                  r'- \mu_{\dot{g}_\text{abl}})^2\right]}$',
                  fontsize=26, y=1.05)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('figures/contours_{grad}.pdf'.format(grad=grad),  bbox_inches='tight')
    plt.close(fig)
