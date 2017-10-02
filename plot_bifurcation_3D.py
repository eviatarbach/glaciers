import numpy

import plot_config

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from data import eq_volume


def c0(G, P):
    res = eq_volume(G, P)
    return res[0] if (len(res) >= 1) else numpy.nan


def c1(G, P):
    res = eq_volume(G, P)
    return res[1] if (len(res) >= 2) else numpy.nan


def c2(G, P):
    res = eq_volume(G, P)
    return res[2] if (len(res) == 3) else numpy.nan


c0v = numpy.vectorize(c0)
c1v = numpy.vectorize(c1)
c2v = numpy.vectorize(c2)

Gmin, Gmax = -0.99, 1.0
Pmin, Pmax = -50, 10

G = numpy.linspace(Gmin, Gmax, 30)
P_neg = numpy.linspace(Pmin, 0, 25)
P_pos = numpy.linspace(0, Pmax, 5)

G_n, P_n = numpy.meshgrid(G, P_neg)
G_p, P_p = numpy.meshgrid(G, P_pos)

Z0_n = c0v(G_n, P_n)
Z1_n = c1v(G_n, P_n)

Z0_p = c0v(G_p, P_p)
Z1_p = c1v(G_p, P_p)
Z2_p = c2v(G_p, P_p)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(G_n, P_n, Z0_n, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')
ax.plot_wireframe(G_n, P_n, Z1_n, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.8)

ax.plot_wireframe(G_p, P_p, Z0_p, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)
ax.plot_wireframe(G_p, P_p, Z1_p, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')
ax.plot_wireframe(G_p, P_p, Z2_p, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)

plt.rc('text', usetex=True)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.set_xlabel('$G^*$', fontsize=18)
ax.set_ylabel('$P^*$', fontsize=18)
ax.set_zlabel('$V_s^*$', fontsize=18)
plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=14)
plt.yticks(fontsize=14)
for t in ax.xaxis.get_major_ticks():
    t.set_pad(-2.5)
for t in ax.yaxis.get_major_ticks():
    t.set_pad(-5)
for t in ax.zaxis.get_major_ticks():
    t.label.set_fontsize(14)
    t.set_pad(2)
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
plt.savefig('figures/bifurcation_3D.pdf')
