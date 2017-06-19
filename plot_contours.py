import pandas
import numpy
from scipy.integrate import dblquad

import plot_config

import matplotlib.pyplot as plt

ERRS = {'g_abl': 0.004487,  # root-mean square error in interpolating g_abl
        'g_acc': 0.001889}  # root-mean square error in interpolating g_acc

glaciers = pandas.read_pickle('data/serialized/glaciers_climate')

g_abl_pts = numpy.linspace(glaciers['g_abl'].min(), glaciers['g_abl'].max(), 25)
g_acc_pts = numpy.linspace(glaciers['g_acc'].min(), glaciers['g_acc'].max(), 25)

a = (glaciers['g_acc']/glaciers['g_abl']).min()
b = (glaciers['g_acc']/glaciers['g_abl']).max()

mu_x = numpy.mean(glaciers['g_acc'])
mu_y = numpy.mean(glaciers['g_abl'])


def f(mu_x, mu_y, x, y):
    std_x, std_y = ERRS['g_acc'], ERRS['g_abl']
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


Z = std_acc(X, Y)
cp = plt.contourf(X, Y, Z, cmap='Blues_r')
cbar = plt.colorbar(cp)
cbar.ax.tick_params(labelsize=14)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.xlabel('$\dot{g}_\mathrm{acc}$ $\left(y^{-1}\\right)$', fontsize=22)
plt.ylabel('$\dot{g}_\mathrm{abl}$ $\left(y^{-1}\\right)$', fontsize=22, rotation=0, labelpad=45)
plt.title('$\sqrt{E\left[(\dot{g}_\mathrm{acc} - \mu_{\dot{g}_\mathrm{acc}})^2\\right]}$',
          fontsize=22, y=1.03)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('figures/contours_acc.pdf',  bbox_inches='tight')
plt.clf()

Z = std_abl(X, Y)
cp = plt.contourf(X, Y, Z, cmap='Blues_r')
cbar = plt.colorbar(cp)
cbar.ax.tick_params(labelsize=14)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.xlabel('$\dot{g}_\mathrm{acc}$ $\left(y^{-1}\\right)$', fontsize=22)
plt.ylabel('$\dot{g}_\mathrm{abl}$ $\left(y^{-1}\\right)$', fontsize=22, rotation=0, labelpad=45)
plt.title('$\sqrt{E\left[(\dot{g}_\mathrm{abl} - \mu_{\dot{g}_\mathrm{abl}})^2\\right]}$',
          fontsize=22, y=1.03)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('figures/contours_abl.pdf',  bbox_inches='tight')
