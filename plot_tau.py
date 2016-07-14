import numpy
import pandas
import matplotlib.pyplot as plt

RGI_NAMES = sorted(['Alaska', 'Western Canada & USA', 'Arctic Canada (North)',
                    'Arctic Canada (South)', 'Greenland (periphery)', 'Iceland',
                    'Svalbard & Jan Mayen', 'Scandinavia', 'Russian Arctic', 'North Asia',
                    'Central Europe', 'Caucasus & Middle East', 'Central Asia',
                    'South Asia (West)', 'South Asia (East)', 'Low Latitudes', 'Southern Andes',
                    'New Zealand'])

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

sens = all_glaciers[['tau_weighted', 'tau_gz_weighted', 'tau_median',
                     'tau_gz_median', 'tau_mid', 'tau_gz_mid']]

means = sens.groupby(level='Region').mean()

sens_means = (means['tau_weighted'] + means['tau_median']
              + means['tau_mid'])/3

order = numpy.argsort(sens_means.values)

indices = numpy.arange(18)

plt.plot(means['tau_weighted'][order], indices[range(18)] - 0.2, 'o',
         markerfacecolor='black', markersize=8)
plt.plot(means['tau_median'][order], indices[range(18)], 'o', markerfacecolor='white',
         markeredgewidth=3)
plt.plot(means['tau_mid'][order], indices[range(18)] + 0.2, 'o', markerfacecolor='white',
         markeredgewidth=2, markersize=7)

plt.hlines(indices, [0], 300, linestyles='dotted')

plt.yticks(range(18), numpy.array(RGI_NAMES)[order], horizontalalignment='left', fontsize=14)
plt.xticks(fontsize=12)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Optima']

ax = plt.gca()
ax.tick_params(axis='y', direction='out', pad=155, length=0)
ax.tick_params(axis='x', direction='out')

ax.set_xlim([0, 250])
ax.set_ylim([-1, 18])

plt.xlabel('Mean $e$-folding time (years)', fontsize=16)

plt.legend(['Area-weighted', 'Median', 'Mid-range'], numpoints=1, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 1), frameon=False, fontsize=14)

plt.show()
