import PyDSTool as dst
import numpy as np
from matplotlib import pyplot as plt

# we must give a name
DSargs = dst.args(name='Block model with scaling')
# parameters
DSargs.pars = {'g': 1, 'P': 0, 'alpha': 8/11., 'beta': 13/11.}
# rhs of the differential equation
DSargs.varspecs = {'V': 'g*(-V^(alpha)*P - V^(beta) + V)',
                   'w': 'V-w' }
# initial conditions
DSargs.ics = {'V': 0, 'w': 0 }

ode  = dst.Generator.Vode_ODEsystem(DSargs)

ode.set(pars={'P': 0.1859})
ode.set(ics={'V': 0.0602326})

PC = dst.ContClass(ode)            # Set up continuation class

PCargs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars     = ['P']                    # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.MaxNumPoints = 1500                      # The following 3 parameters are set after trial-and-error
PCargs.MaxStepSize  = 2e-4
PCargs.MinStepSize  = 1e-5
PCargs.StepSize     = 2e-4
PCargs.LocBifPoints = 'LP'                     # detect limit points / saddle-node bifurcations
PCargs.SaveEigen    = True                     # to tell unstable from stable branches

PC.newCurve(PCargs)
PC['EQ1'].forward()

PCargs = dst.args(name='EQ2', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars     = ['P']                    # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.initpoint = 'EQ1:P1'
PCargs.MaxNumPoints = 400                      # The following 3 parameters are set after trial-and-error
PCargs.MaxStepSize  = 2e-4
PCargs.MinStepSize  = 1e-5
PCargs.StepSize     = 2e-4
PCargs.SaveEigen    = True                     # to tell unstable from stable branches

PC.newCurve(PCargs)
PC['EQ2'].backward()
PC.display(['P','V'], stability=True, figure=3, points=False)        # stable and unstable branches as solid and dashed curves, resp.

import matplotlib
matplotlib.rc('text', usetex=True)
plt.figure(3)
plt.plot([0.12, 0.19], [0, 0], 'black')
plt.xlabel('$P^*$', fontsize=16)
plt.ylabel('$V^*$', fontsize=16, rotation=0)
plt.title('')
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
