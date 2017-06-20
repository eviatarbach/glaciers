import plot_config

import matplotlib.pyplot as plt
import pandas

sens_HSIC = pandas.read_table('data/HSIC_sens.txt', sep=',')

ax = plt.subplot(111)
plt.plot(sens_HSIC['original'] - sens_HSIC['bias'], range(7), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)
plt.hlines(range(7), 0, 1, linestyles='dotted', linewidth=1.5)
plt.hlines(range(7), sens_HSIC['min. c.i.'], sens_HSIC['max. c.i.'], linewidth=2.5)

ax.set_xlim([0, 1])
ax.set_ylim([-1, 7])

plt.yticks(range(7), ['$G^*$', '$z_\mathrm{ela}$', '$c_a$', '$c_l$', r'$\beta$', '$V$',
                      '$\Gamma$'], fontsize=18, horizontalalignment='left')
plt.xticks(fontsize=16)

yax = ax.get_yaxis()
yax.set_tick_params(pad=35)

plt.xlabel('Normalized HSIC for sensitivity', fontsize=18)

fig = plt.gcf()
fig.set_size_inches(9, 3.5)
plt.tight_layout()

plt.savefig('figures/HSIC_sens.pdf')
plt.clf()

tau_HSIC = pandas.read_table('data/HSIC_tau.txt', sep=',')

ax = plt.subplot(111)
plt.plot(tau_HSIC['original'] - tau_HSIC['bias'], range(7), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)
plt.hlines(range(7), 0, 0.4, linestyles='dotted', linewidth=1.5)
plt.hlines(range(7), tau_HSIC['min. c.i.'], tau_HSIC['max. c.i.'], linewidth=2.5)

ax.set_xlim([0, 0.4])
ax.set_ylim([-1, 7])

plt.yticks(range(7), ['$\dot{g}_\mathrm{acc}$', '$\dot{g}_\mathrm{abl}$', '$z_\mathrm{ela}$',
                      '$c_a$', '$c_l$', r'$\beta$', '$V$'], fontsize=18,
           horizontalalignment='left')
plt.xticks(fontsize=16)

yax = ax.get_yaxis()
yax.set_tick_params(pad=35)

plt.xlabel('Normalized HSIC for response time', fontsize=18)

fig = plt.gcf()
fig.set_size_inches(9, 3.5)
plt.tight_layout()

plt.savefig('figures/HSIC_tau.pdf')
