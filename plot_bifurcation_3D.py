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

Gmin, Gmax = -1.2, 1.0
Pmin, Pmax = -50, 10

G = numpy.linspace(Gmin, Gmax, 30)
G_neg = numpy.linspace(Gmin, 0, 15)
G_pos = numpy.linspace(0, Gmax, 15)
P_neg = numpy.linspace(Pmin, 0, 25)
P_pos = numpy.linspace(0, Pmax, 5)

G_an, P_an = numpy.meshgrid(G, P_neg)
G_nn, P_nn = numpy.meshgrid(G_neg, P_neg)
G_pp, P_pp = numpy.meshgrid(G_pos, P_pos)
G_pn, P_pn = numpy.meshgrid(G_pos, P_neg)
G_np, P_np = numpy.meshgrid(G_neg, P_pos)

Z1_an = c1v(G_an, P_an)

Z0_nn = c0v(G_nn, P_nn)

Z0_pn = c0v(G_pn, P_pn)

Z0_np = c0v(G_np, P_np)
Z1_np = c1v(G_np, P_np)
Z2_np = c2v(G_np, P_np)

Z0_pp = c0v(G_pp, P_pp)
Z1_pp = c1v(G_pp, P_pp)
Z2_pp = c2v(G_pp, P_pp)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(G_an, P_an, Z1_an, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.8)

ax.plot_wireframe(G_nn, P_nn, Z0_nn, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.8)

ax.plot_wireframe(G_pn, P_pn, Z0_pn, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')

ax.plot_wireframe(G_np, P_np, Z0_np, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)
ax.plot_wireframe(G_np, P_np, Z1_np, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')
ax.plot_wireframe(G_np, P_np, Z2_np, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)

ax.plot_wireframe(G_pp, P_pp, Z0_pp, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)
ax.plot_wireframe(G_pp, P_pp, Z1_pp, alpha=0.5, cstride=1, rstride=1, color='black', linewidth=0.7,
                  linestyle='dotted')
ax.plot_wireframe(G_pp, P_pp, Z2_pp, alpha=0.5, cstride=1, rstride=1, color='black',
                  linewidth=0.8)

plt.rc('text', usetex=True)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.set_xlabel('$G^*$', fontsize=18)
ax.set_ylabel('$P^*$', fontsize=18)
ax.set_zlabel('$V_s^*$', fontsize=18)
plt.xticks(fontsize=14)
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
