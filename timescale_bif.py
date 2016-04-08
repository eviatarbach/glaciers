import numpy
import matplotlib.pyplot as plt

params = numpy.vstack([6/125*numpy.sqrt(15) + numpy.power(10*numpy.ones([1, 6]), numpy.arange(-6, 0, 1)), 0.0602326*numpy.ones(6)])

dt = 0.1
T = 10000
t = numpy.arange(0, T, dt)
g = 1

alpha = 8/11
beta = 13/11

V = numpy.zeros([params.shape[1], len(t)], dtype='complex128')  # need to use complex due to precision issues

V[:, 0] = params[1, :]

for i in range(1, len(t)):
    V[:, i] = V[:, i - 1] + dt*g*(-params[0, :]*V[:, i - 1]**alpha - V[:, i - 1]**beta + V[:, i - 1])

for i, curve in enumerate(V):
    plt.semilogx(t, curve, label='${:.5f}$'.format(params[0, i]))

plt.semilogx(t, numpy.ones(len(t))*0.0602326/numpy.e, color='black', linestyle='--')

import matplotlib
matplotlib.rc('text', usetex=True)
legend = plt.legend(title='$P^*$', fontsize=14, loc='lower left', bbox_to_anchor=(0, 0), frameon=False)
plt.setp(legend.get_title(), fontsize=16)
plt.xlabel('$t^*$', fontsize=16)
plt.ylabel('$V^*$', fontsize=16, rotation=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()
