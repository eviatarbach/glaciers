"""
Need to increase memory to compile figures, see
https://olezfdtd.wordpress.com/2010/03/16/extending-latex-memory/
"""

import pickle

import pandas

import plot_config

import matplotlib.pyplot as plt

LABEL_SIZE = 22
TICK_SIZE = 18

all_data = pickle.load(open('all_data', 'rb'))
sens = sum([d['sensitivity'] for d in all_data])/len(all_data)
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')
slopes = all_glaciers.loc[sens.index]['Slope']
vols = all_glaciers.loc[sens.index]['volume']
vols_ss = sum([d['volumes_ss'] for d in all_data])/len(all_data)

df_concat = pandas.concat([d['tau'] for d in all_data])
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()

# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# Tau vs. slope
plt.hexbin(slopes, df_means, yscale='log', cmap='Blues_r', bins='log', gridsize=(40, 15))
plt.title('$e$-folding time (years)', fontsize=LABEL_SIZE)
plt.xlabel('Slope (rad)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax = plt.gca()
ax.set_xlim([0, max(slopes[~df_means.isnull()])])
ax.set_ylim([0, max(df_means)])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/tau_slope.pdf')
plt.clf()

# Tau vs. volume
plt.hexbin(vols, df_means, xscale='log', yscale='log', cmap='Blues_r', bins='log')
plt.title('$e$-folding time (years)', fontsize=LABEL_SIZE)
plt.xlabel('Volume (m$^3$)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax = plt.gca()
ax.set_xlim([0, max(vols)])
ax.set_ylim([0, max(df_means)])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/tau_vol.pdf')
plt.clf()

# Sensitivity vs. volume
index = sens < 0
plt.hexbin(vols[index], (-sens/vols_ss)[index], xscale='log', yscale='log', cmap='Blues_r',
           bins='log')
plt.title('Normalized sensitivity (K$^{-1}$)', fontsize=LABEL_SIZE)
plt.xlabel('Volume (m$^3$)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax = plt.gca()
ax.set_xlim([0, max(vols[index])])
ax.set_ylim([0, max((-sens/vols_ss)[index])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/sens_vol.pdf')
plt.clf()

# Sensitivity vs. slopes
index2 = sens < 0
plt.hexbin(slopes[index2], (-sens/vols_ss)[index2], yscale='log', cmap='Blues_r', bins='log')
plt.title('Normalized sensitivity (K$^{-1}$)', fontsize=LABEL_SIZE)
plt.xlabel('Slope (rad)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax = plt.gca()
ax.set_xlim([0, max(slopes[index2])])
ax.set_ylim([0, max((-sens/vols_ss)[index2])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/sens_slope.pdf')
