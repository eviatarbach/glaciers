import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data import eq_volume
import plot_config
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
Pmin, Pmax = -50, 10

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

ax.plot_wireframe(Gmesh, Pmesh, Z2, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.8)
ax.plot_wireframe(Gmesh, Pmesh, Z1, alpha=1, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')
ax.plot_wireframe(Gmesh_neg, Pmesh_neg, Z0_neg, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)
plt.rc('text', usetex=True)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.set_xlabel('$G^*$', fontsize=22)
ax.set_ylabel('$P^*$', fontsize=22)
ax.set_zlabel('$V_s^*$', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for t in ax.xaxis.get_major_ticks():
    t.set_pad(-2.5)
for t in ax.yaxis.get_major_ticks():
    t.set_pad(-5)
for t in ax.zaxis.get_major_ticks():
    t.label.set_fontsize(18)
    t.set_pad(2)
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
plt.show()
