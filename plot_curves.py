import numpy

import plot_config

import matplotlib.pyplot as plt

from data import p, gamma

P = numpy.arange(0.36, 0.4, 0.008)

dt = 0.001
T = 300
t = numpy.arange(0, T, dt)

V = numpy.zeros([len(P), len(t)],
                dtype='complex128')  # need to use complex due to precision issues

V[:, 0] = 0.1

for i in range(1, len(t)):
    V[:, i] = V[:, i - 1] + dt*(V[:, i - 1] - P*V[:, i - 1]**(1/gamma)
                                - V[:, i - 1]**(1/gamma + 1/p))

plt.figure(figsize=(10, 5))
for i, curve in enumerate(V):
    plt.plot(t, curve, label='${:.3f}$'.format(P[i]))

legend = plt.legend(title='$P^*$', fontsize=18, bbox_to_anchor=(1.35, 1.0))
plt.setp(legend.get_title(), fontsize=18)
plt.xlabel('$t^*$', fontsize=22)
plt.ylabel('$V^*$', fontsize=22, rotation=0, labelpad=25)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlim([0, 300])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

plt.savefig('figures/curves.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
