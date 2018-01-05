import pickle

import numpy
import pandas
from uncertainties import unumpy

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

single_data = pandas.read_pickle('data/serialized/single_data')

means = numpy.exp(numpy.log(single_data['tau'][single_data['tau'] > 0]).groupby(level='Region').mean())

means = []
stds = []
for region in RGI_REGIONS:
    tau = unumpy.uarray(single_data.loc[region]['tau'], single_data.loc[region]['tau_std'])
    region_tau = unumpy.exp(unumpy.log(single_data.loc[region]['tau'][single_data.loc[region]['tau'] > 0]).mean())
    means.append(unumpy.nominal_values(region_tau))
    stds.append(unumpy.std_devs(region_tau))
means = numpy.array(means)
stds = numpy.array(stds)

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(means[::-1], range(19), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)

plt.hlines(indices, 0, means[::-1], linestyles='dotted', linewidth=1.5)
plt.hlines(indices, (means - stds)[::-1], (means + stds)[::-1], linewidth=2.5)

plt.yticks(range(19), RGI_NAMES[::-1], fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(left='off', labelleft='off')

ax.set_xlim([0, 185])
ax.set_ylim([-1, 19])

plt.xlabel('Geometric mean $e$-folding time (years)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(9, 7)
plt.tight_layout()

plt.savefig('figures/tau.pdf')
