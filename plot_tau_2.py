import pickle

import numpy
import pandas
import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS


def replace(group, stds=3):
    group2 = group.copy()
    group2[numpy.abs(group2 - group2.mean()) > stds*group2.std()] = numpy.nan
    return group2

all_data = pickle.load(open('all_data', 'rb'))
tau = [d['tau'] for d in all_data]
rel = [t.groupby(level='Region').mean() for t in tau]
concat = pandas.concat(rel, axis=1).apply(replace)
means = concat.mean(axis=1)
stds = concat.std(axis=1)

indices = numpy.arange(19)

ax = plt.subplot(111)

plt.plot(means[RGI_REGIONS], range(19), 'o', markerfacecolor='black', markersize=8)

plt.hlines(indices, 0, 200, linestyles='dotted', linewidth=1.5)
plt.hlines(indices, (means - stds)[RGI_REGIONS], (means + stds)[RGI_REGIONS], linewidth=2.5)

plt.yticks(range(19), RGI_NAMES, fontsize=22, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=245)

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Optima']
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

ax.set_xlim([0, 200])
ax.set_ylim([-1, 19])

plt.xlabel('Mean $e$-folding time (years)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.tight_layout()

plt.savefig('/home/eviatar/Documents/glaciers/glacier_poster/figures/tau.png')
