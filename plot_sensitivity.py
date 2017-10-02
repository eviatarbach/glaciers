import pickle

import numpy
import pandas

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

all_data = pickle.load(open('data/serialized/all_data', 'rb'))
single_data = pandas.read_pickle('data/serialized/single_data')
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

means_single = (-(single_data['sensitivity']).groupby(level='Region').sum()
                /single_data.groupby(level='Region')['volumes_ss'].sum())

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(means_single[RGI_REGIONS[::-1]], range(19), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)

plt.hlines(indices, 0, means_single[RGI_REGIONS[::-1]], linestyles='dotted', linewidth=1.5)

plt.yticks(range(19), RGI_NAMES[::-1], fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=245)

ax.set_xlim([0, 0.008])
ax.set_ylim([-1, 19])

plt.xlabel('Normalized sensitivity to ELA (m$^{-1}$)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.tight_layout()

plt.savefig('figures/sens.pdf')
