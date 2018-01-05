import plot_config

import matplotlib.pyplot as plt
import pandas

sens_HSIC = pandas.read_table('data/HSIC_sens.txt', sep=',')

ax = plt.subplot(111)
plt.plot(sens_HSIC['original'] - sens_HSIC['bias'], range(5), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)
plt.hlines(range(5), 0, 1, linestyles='dotted', linewidth=1.5)
plt.hlines(range(5), sens_HSIC['min. c.i.'], sens_HSIC['max. c.i.'], linewidth=2.5)

ax.set_xlim([0, 1])
ax.set_ylim([-1, 5])

plt.yticks(range(6), ['$G^*$', 'log $c_a$', 'log $c_l$', r'tan${}^{-1}\beta$', 'log $V$'],
           fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=70)

plt.xlabel('$S_i$ (sensitivity)', fontsize=20)

fig = plt.gcf()
fig.set_size_inches(9, 3.5)
plt.tight_layout()

plt.savefig('figures/HSIC_sens.pdf')
plt.clf()

tau_HSIC = pandas.read_table('data/HSIC_tau.txt', sep=',')

ax = plt.subplot(111)
plt.plot(tau_HSIC['original'] - tau_HSIC['bias'], range(6), 'o', markerfacecolor='black',
         markeredgecolor='black', markersize=8)
plt.hlines(range(6), 0, 0.45, linestyles='dotted', linewidth=1.5)
plt.hlines(range(6), tau_HSIC['min. c.i.'], tau_HSIC['max. c.i.'], linewidth=2.5)

ax.set_xlim([0, 0.45])
ax.set_ylim([-1, 6])

plt.yticks(range(6), [r'$G^*$', r'$\dot{g}_\text{abl}$', 'log $c_a$', 'log $c_l$',
                      r'tan${}^{-1}\beta$', 'log $V$'], fontsize=20, horizontalalignment='left')
plt.xticks(fontsize=18)

yax = ax.get_yaxis()
yax.set_tick_params(pad=70)

plt.xlabel('$S_i$ (response time)', fontsize=20)

fig = plt.gcf()
fig.set_size_inches(9, 3.5)
plt.tight_layout()

plt.savefig('figures/HSIC_tau.pdf')
