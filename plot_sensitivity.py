import numpy
import pandas
import matplotlib.pyplot as plt

RGI_NAMES = ['Alaska', 'Western Canada & USA', 'Arctic Canada (North)', 'Arctic Canada (South)',
             'Greenland (periphery)', 'Iceland', 'Svalbard & Jan Mayen', 'Scandinavia',
             'Russian Arctic', 'North Asia', 'Central Europe', 'Caucasus & Middle East',
             'Central Asia', 'South Asia (West)', 'South Asia (East)', 'Low Latitudes',
             'Southern Andes', 'New Zealand', 'Antarctic & Subantarctic']

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

sens = -(all_glaciers[['sensitivity_weighted', 'sensitivity_gz_weighted', 'sensitivity_median',
                       'sensitivity_gz_median', 'sensitivity_mid', 'sensitivity_gz_mid']]
         .divide(all_glaciers['volume'], axis=0))

sens = sens.replace(0, numpy.nan)

means = sens.groupby(level='Region').mean()

indices = numpy.hstack([numpy.arange(3*i + 2*i, 3*i + 2*i + 3) for i in range(18)])

plt.plot(means['sensitivity_weighted'], indices[range(0, 18*3, 3)], 'o', markerfacecolor='black',
         markersize=8)
plt.plot(means['sensitivity_median'], indices[range(1, 18*3, 3)], 'o', markerfacecolor='white',
         markeredgewidth=3)
plt.plot(means['sensitivity_mid'], indices[range(2, 18*3, 3)], 'o', markerfacecolor='white',
         markeredgewidth=2, markersize=7)

plt.hlines(indices, [0], numpy.hstack([means[['sensitivity_weighted']],
                                       means[['sensitivity_median']],
                                       means[['sensitivity_mid']]]).flatten(),
           linestyles='dotted')

plt.yticks(list(indices) + [3*i + 2*i - 2 for i in range(1, 19)],
           ['    Area-weighted', '    Median', '    Mid-range']*18
           + list(map(str.upper, RGI_NAMES)), horizontalalignment='left')

ax = plt.gca()
ax.tick_params(axis='y', direction='out', pad=165, length=0)

plt.xlabel('Mean normalized sensitivity to ELA perturbations (m$^{-1}$)')

plt.show()
