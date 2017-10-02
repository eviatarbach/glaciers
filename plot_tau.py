import pickle

import numpy
import pandas

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

RGI_REGIONS = RGI_REGIONS[::-1]
RGI_NAMES = RGI_NAMES[::-1]

all_data = pickle.load(open('data/serialized/all_data', 'rb'))
single_data = pandas.read_pickle('data/serialized/single_data')

tau = [numpy.log(d['tau'][d['tau'] > 0]) for d in all_data]

means_single = numpy.exp(numpy.log(single_data['tau'][single_data['tau'] > 0]).groupby(level='Region').mean())

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(means_single[RGI_REGIONS], range(19), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)

plt.hlines(indices, 0, means_single[RGI_REGIONS], linestyles='dotted', linewidth=1.5)

# plt.yticks(range(19), RGI_NAMES, fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(left='off', labelleft='off')

ax.set_xlim([0, 135])
ax.set_ylim([-1, 19])

plt.xlabel('Geometric mean $e$-folding time (years)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(9, 7)
plt.tight_layout()

plt.savefig('figures/tau.pdf')
