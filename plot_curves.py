import numpy
import matplotlib.pyplot as plt

params = numpy.vstack([numpy.arange(0.36, 0.4, 0.005), 0.1*numpy.ones(9)])

dt = 0.001
T = 300
t = numpy.arange(0, T, dt)
g = 1

alpha = 4/5.
beta = 7/5.

V = numpy.zeros([params.shape[1], len(t)], dtype='complex128')  # need to use complex due to precision issues

V[:, 0] = params[1, :]

for i in range(1, len(t)):
    V[:, i] = V[:, i - 1] + dt*g*(-params[0, :]*V[:, i - 1]**alpha - V[:, i - 1]**beta + V[:, i - 1])

for i, curve in enumerate(V):
    plt.plot(t, curve, label='${:.3f}$'.format(params[0, i]))

import matplotlib
matplotlib.rc('text', usetex=True)
legend = plt.legend(title='$P^*$', fontsize=18, bbox_to_anchor=(1.35, 1.0))
plt.setp(legend.get_title(), fontsize=18)
plt.xlabel('$t^*$', fontsize=22)
plt.ylabel('$V^*$', fontsize=22, rotation=0)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

plt.show()
