import pandas
import numpy
import scipy.optimize

import plot_config

import matplotlib.pyplot as plt

all_glaciers = pandas.read_pickle('data/serialized/all_glaciers')

beta_H = numpy.arctan(all_glaciers['SLOPE_avg'])
beta_R = numpy.arctan(all_glaciers['Slope'])

diff = beta_H - beta_R

intervals = numpy.logspace(numpy.log10(beta_H.min()), numpy.log10(beta_H.max()), 50)

mid = (intervals[:-1] + intervals[1:])/2
err = numpy.zeros(len(mid))

for i, _ in enumerate(intervals[:-1]):
    mask = ((beta_H > intervals[i]) & (beta_H < intervals[i + 1]) & (diff > 0))
    err[i] = numpy.sqrt((diff[mask]**2).mean())

c, r = scipy.optimize.curve_fit(lambda slope, c, r: c*slope**r, xdata=mid[~numpy.isnan(err)],
                                ydata=err[~numpy.isnan(err)])[0]

plt.scatter(numpy.log10(mid), err)

slope = numpy.linspace(mid[(~numpy.isnan(err)).nonzero()[0][0]], max(mid), 100)
plt.plot(numpy.log10(slope), c*slope**r)
plt.xlabel(r'$\text{log}_{10}(\text{tan}^{-1}(\beta_H))$', fontsize=18)
plt.ylabel('RMSE', fontsize=18, rotation=0, labelpad=45)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('figures/err.pdf', bbox_inches='tight')
