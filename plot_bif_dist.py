import pickle

import numpy
import pandas

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

single_data = pandas.read_pickle('data/serialized/single_data')

bif_dists = []
for region in RGI_REGIONS:
    bif_dist = single_data.loc[region]['bif_dist']
    vols = single_data.loc[region]['volumes_ss']
    mask = vols > 0
    bif_dists.append(sum(vols[mask]*bif_dist[mask])/sum(vols))
bif_dists = numpy.array(bif_dists)

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(bif_dists[::-1], range(19), 'o-', markerfacecolor='black',
         markeredgecolor='black', markersize=8)

plt.hlines(indices, 0, bif_dists[::-1], linestyles='dotted', linewidth=1.5)

plt.yticks(range(19), RGI_NAMES[::-1], fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(left='off', labelleft='off')

ax.set_xlim([0, 1700])
ax.set_ylim([-1, 19])

plt.xlabel('ELA distance (m)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(9, 7)
plt.tight_layout()

plt.savefig('figures/bif_dist.pdf')
