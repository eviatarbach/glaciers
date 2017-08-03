import numpy

from data import p, gamma

import plot_config

import matplotlib.pyplot as plt

P0 = -((gamma - 1)*p - gamma)/((gamma - 1)*((gamma - 1)*p/gamma)**(gamma/(gamma*p - gamma - p))*p)

V = numpy.linspace(0, 0.14, 500)

labels = ['0.37', '0.38', '$P_0^*$', '0.39']
for i, P in enumerate([0.37, 0.38, P0, 0.39]):
    plt.plot(V, V - P*V**(1/gamma) - V**(1/gamma + 1/p), label=labels[i])

legend = plt.legend(title='$P^*$', fontsize=20, bbox_to_anchor=(1.12, 1.0))
plt.setp(legend.get_title(), fontsize=18)
plt.xlabel('$V^*$', fontsize=24)
plt.ylabel('$F(0, P^*, V^*)$', fontsize=24, rotation=0, labelpad=53)
plt.plot([0, 0.15], [0, 0], color='black', linewidth=0.6, zorder=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlim([0, 0.14])
ax.set_ylim([-0.003, 0.003])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('figures/roots.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
