import pandas
import numpy
import scipy.optimize

import plot_config

import matplotlib.pyplot as plt

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

diff = all_glaciers['SLOPE_avg'] - all_glaciers['Slope']

intervals = numpy.logspace(numpy.log10(all_glaciers['SLOPE_avg'].min()),
                           numpy.log10(all_glaciers['SLOPE_avg'].max()), 50)

mid = (intervals[:-1] + intervals[1:])/2
err = numpy.zeros(len(mid))

for i, _ in enumerate(intervals[:-1]):
    mask = ((all_glaciers['SLOPE_avg'] > intervals[i])
            & (all_glaciers['SLOPE_avg'] < intervals[i + 1]) & (diff > 0))
    err[i] = numpy.sqrt((diff[mask]**2).mean())

c, r = scipy.optimize.curve_fit(lambda slope, c, r: c*slope**r, xdata=mid[~numpy.isnan(err)],
                                ydata=err[~numpy.isnan(err)])[0]

plt.scatter(mid, err)

slope = numpy.linspace(0, max(mid), 100)
plt.plot(slope, c*slope**r)
plt.xlabel(r'$\beta_H$', fontsize=18)
plt.ylabel('RMSE', fontsize=18, rotation=0, labelpad=45)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('figures/err.pdf', bbox_inches='tight')
