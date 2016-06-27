import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data import eq_volume
# TODO: negative G, positive P


def c0(G, P):
    res = eq_volume(G, P)
    return res[0] if (len(res) >= 1) else numpy.nan


def c1(G, P):
    res = eq_volume(G, P)
    if G < 0:
        return res[1] if (len(res) >= 2) else numpy.nan
    else:
        return res[0]


def c2(G, P):
    res = eq_volume(G, P)
    if G < 0:
        return res[2] if (len(res) >= 3) else numpy.nan
    else:
        return res[1] if (len(res) >= 2) else numpy.nan

c0v = numpy.vectorize(c0)
c1v = numpy.vectorize(c1)
c2v = numpy.vectorize(c2)

Gmin, Gmax = -1.2, 1.0
Pmin, Pmax = -45, 10

G = numpy.linspace(Gmin, Gmax, 30)
G_neg = numpy.linspace(Gmin, 0, 15)
G_pos = numpy.linspace(0, Gmax, 15)
P = numpy.linspace(Pmin, Pmax, 30)

Gmesh, Pmesh = numpy.meshgrid(G, P)
Gmesh_neg, Pmesh_neg = numpy.meshgrid(G_neg, P)
Gmesh_pos, Pmesh_pos = numpy.meshgrid(G_pos, P)

Z0_neg = c0v(Gmesh_neg, Pmesh_neg)

Z1 = c1v(Gmesh, Pmesh)
Z2 = c2v(Gmesh, Pmesh)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(Gmesh, Pmesh, Z2, alpha=0.5, cstride=1, rstride=1, color='black')
ax.plot_wireframe(Gmesh, Pmesh, Z1, alpha=1, cstride=1, rstride=1, color='black',
                  linestyle='dashdot')
ax.plot_wireframe(Gmesh_neg, Pmesh_neg, Z0_neg, alpha=0.5, cstride=1, rstride=1, color='black')
