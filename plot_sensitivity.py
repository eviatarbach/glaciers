import pickle

import numpy
import pandas
from uncertainties import unumpy

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

single_data = pandas.read_pickle('data/serialized/single_data')
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

single_data = single_data[-(single_data['sensitivity']/single_data['volumes_ss']) < 1]

means = []
stds = []
for region in RGI_REGIONS:
    sens = unumpy.uarray(single_data.loc[region]['sensitivity'],
                         single_data.loc[region]['sensitivity_std'])
    vols = unumpy.uarray(single_data.loc[region]['volumes_ss'],
                         single_data.loc[region]['volumes_ss_std'])
    mask = ~unumpy.isnan(sens)
    region_sens = sum(-sens[mask])/sum(vols[mask])
    means.append(unumpy.nominal_values(region_sens))
    stds.append(unumpy.std_devs(region_sens))
means = numpy.array(means)
stds = numpy.array(stds)

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(means[::-1], range(19), 'o', markerfacecolor='black', markeredgecolor='black',
         markersize=8)

plt.hlines(indices, 0, means[::-1], linestyles='dotted', linewidth=1.5)
plt.hlines(indices, (means - stds)[::-1], (means + stds)[::-1], linewidth=2.5)

plt.yticks(range(19), RGI_NAMES[::-1], fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=245)

ax.set_xlim([0, 0.020])
ax.set_ylim([-1, 19])

plt.xlabel('Normalized sensitivity to ELA (m$^{-1}$)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.tight_layout()

plt.savefig('figures/sens.pdf')
