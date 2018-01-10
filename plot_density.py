import numpy
import pandas

import plot_config

import matplotlib.pyplot as plt
import mpl_scatter_density

# Make the norm object to define the image stretch
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from data import gamma

LABEL_SIZE = 18
TICK_SIZE = 14

norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())

single_data = pandas.read_pickle('data/serialized/single_data')
sens = single_data['sensitivity']
all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')
slopes = numpy.arctan(all_glaciers['SLOPE_avg'])
vols = all_glaciers['volume']
vols_ss = single_data['volumes_ss']

tau = single_data['tau']

# Tau vs. slope
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(slopes[tau > 0], tau[tau > 0], cmap='Blues_r', norm=norm, dpi=40)
plt.yscale('log')
plt.title('$e$-folding time (years)', fontsize=LABEL_SIZE)
plt.xlabel('Slope (rad)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax.set_xlim([min(slopes[tau > 0]), max(slopes[tau > 0])])
ax.set_ylim([min(tau[tau > 0]), max(tau[tau > 0])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/tau_slope.pdf')
plt.clf()

# Tau vs. volume
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(vols[tau > 0], tau[tau > 0], cmap='Blues_r', norm=norm, dpi=40)
plt.xscale('log')
plt.yscale('log')
plt.title('$e$-folding time (years)', fontsize=LABEL_SIZE)
plt.xlabel('Volume (m$^3$)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax.set_xlim([min(vols[tau > 0]), max(vols[tau > 0])])
ax.set_ylim([min(tau[tau > 0]), max(tau[tau > 0])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/tau_vol.pdf')
plt.clf()

# Sensitivity vs. volume
index = sens < 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(vols[index], (-sens/vols_ss)[index], cmap='Blues_r', norm=norm, dpi=40)
plt.xscale('log')
plt.yscale('log')
plt.title('Normalized sensitivity (m$^{-1}$)', fontsize=LABEL_SIZE)
plt.xlabel('Volume (m$^3$)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax.set_xlim([min(vols[index]), max(vols[index])])
ax.set_ylim([min((-sens/vols_ss)[index]), max((-sens/vols_ss)[index])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/sens_vol.pdf')
plt.clf()

# Sensitivity vs. slopes
index2 = sens < 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(slopes[index2], (-sens/vols_ss)[index2], cmap='Blues_r', norm=norm, dpi=40)
plt.yscale('log')
plt.title('Normalized sensitivity (m$^{-1}$)', fontsize=LABEL_SIZE)
plt.xlabel('Slope (rad)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax.set_xlim([min(slopes[index2]), max(slopes[index2])])
ax.set_ylim([min((-sens/vols_ss)[index2]), max((-sens/vols_ss)[index2])])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/sens_slope.pdf')
plt.clf()

# tau vs. tau_V
Ldim = single_data['L0']
volumes_nd = single_data['volumes_ss']/Ldim**3
tau_V = (2*volumes_nd**((3 - 2*gamma)/gamma) + single_data['P']*volumes_nd**(-(gamma - 1)/gamma)
         - 1)**(-1)*all_glaciers['g_abl']**(-1)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(tau[tau > 0], tau_V[tau > 0], cmap='Blues_r', norm=norm)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\tau_V$ (y)', fontsize=LABEL_SIZE, rotation=0, labelpad=30)
plt.xlabel(r'$\tau$ (y)', fontsize=LABEL_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
ax = plt.gca()
ax.set_xlim([min(tau), max(tau)])
ax.set_ylim([min(tau), max(tau)])
ax.get_yaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(which='both', direction='out')
ax.get_xaxis().set_tick_params(top='off')
ax.get_yaxis().set_tick_params(right='off')
plt.tight_layout()
plt.savefig('figures/tau_tau_V.pdf')
