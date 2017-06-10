import pickle

import numpy
import pandas

import plot_config

import matplotlib.pyplot as plt

from data import RGI_NAMES, RGI_REGIONS

RGI_REGIONS = RGI_REGIONS[::-1]
RGI_NAMES = RGI_NAMES[::-1]

all_data = pickle.load(open('all_data', 'rb'))
single_data = pickle.load(open('single_data', 'rb'))

tau = [d['tau'] for d in all_data]
rel = [t.groupby(level='Region').mean() for t in tau]
concat = pandas.concat(rel, axis=1)
# means = concat.mean(axis=1)
stds = concat.std(axis=1)

means_single = single_data.groupby(level='Region')['tau'].mean()

indices = numpy.arange(19)

ax = plt.subplot(111)

# plt.plot(means[RGI_REGIONS], range(19), 'o', markerfacecolor='black', markeredgecolor='black',
#          markersize=8)
plt.plot(means_single[RGI_REGIONS], range(19), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)

plt.hlines(indices, 0, 225, linestyles='dotted', linewidth=1.5)
plt.hlines(indices, (means_single - stds)[RGI_REGIONS], (means_single + stds)[RGI_REGIONS],
           linewidth=2.5)

plt.yticks(range(19), RGI_NAMES, fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=245)

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Optima']
# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

ax.set_xlim([0, 225])
ax.set_ylim([-1, 19])

plt.xlabel('Mean $e$-folding time (years)', fontsize=22)

fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.tight_layout()

plt.savefig('figures/tau.pdf')
