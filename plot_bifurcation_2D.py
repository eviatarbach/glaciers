import numpy

from data import eq_volume

import plot_config
import matplotlib.pyplot as plt

plt.figure(1)
P = numpy.linspace(0.25, 0.4, 5000)
res = [eq_volume(0, P_val) for P_val in P]
plt.plot(P, [p[1] if len(p) == 3 else numpy.nan for p in res], linestyle='--', color='black',
         dashes=(5, 5))
plt.plot(P, [p[2] if len(p) == 3 else numpy.nan for p in res], color='black')
plt.plot([0.25, 0.4], [0, 0], color='black')
plt.xlabel('$P^*$', fontsize=22)
plt.ylabel('$V_s^*$', fontsize=22, rotation=0)
plt.title('')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.set_xlim([0.25, 0.39])
ax.set_ylim([-0.01, 0.234])
plt.tight_layout()
plt.savefig('figures/bifurcation_2D_1.pdf')

plt.figure(2)
P = numpy.linspace(-100, 200, 5000)
res = [eq_volume(0, P_val) for P_val in P]
plt.plot(P, [p[1] if len(p) == 2 else numpy.nan for p in res], color='black')
plt.plot([-100, 0.39], [0, 0], color='black', linestyle='--', dashes=(5, 5))
plt.plot([0.39, 20], [0, 0], color='black')
plt.xlabel('$P^*$', fontsize=22)
plt.ylabel('$V_s^*$', fontsize=22, rotation=0)
plt.title('')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.set_xlim([-100, 20])
plt.show()
plt.tight_layout()
plt.savefig('figures/bifurcation_2D_2.pdf')
