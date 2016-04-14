import PyDSTool as dst
import numpy
from numpy import sign
from matplotlib import pyplot as plt
import scipy.interpolate

# we must give a name
DSargs = dst.args(name='Block model with scaling')
# parameters
DSargs.pars = {'g': 1, 'P': 0.3849, 'alpha': 4/5., 'beta': 7/5.}
# rhs of the differential equation
DSargs.varspecs = {'V': 'g*(-V^(alpha)*P - V^(beta) + V)',
                   'w': 'V-w' }
# initial conditions
DSargs.ics = {'V': 0, 'w': 0 }

ode  = dst.Generator.Vode_ODEsystem(DSargs)

ode.set(pars={'P': 0.3849})
ode.set(ics={'V': 0.06415003})

PC = dst.ContClass(ode)            # Set up continuation class

PCargs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars     = ['P']                    # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.MaxNumPoints = 5000                      # The following 3 parameters are set after trial-and-error
PCargs.MaxStepSize  = 100
PCargs.MinStepSize  = 10
PCargs.StepSize     = 50

PC.newCurve(PCargs)
PC['EQ1'].forward()

curve = PC['EQ1'].curve

P = curve[:, 2][curve[:, 0] >= 0.06415003]
V = curve[:, 0][curve[:, 0] >= 0.06415003]

f1 = scipy.interpolate.interp1d(P, V)

PCargs2 = dst.args(name='EQ2', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs2.freepars     = ['P']                    # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs2.MaxNumPoints = 70000                      # The following 3 parameters are set after trial-and-error
PCargs2.MaxStepSize  = 1e-3
PCargs2.MinStepSize  = 1e-5
PCargs2.StepSize     = 2e-4

PC.newCurve(PCargs2)
PC['EQ2'].forward()

curve2 = PC['EQ2'].curve

P2 = curve2[:, 2][curve2[:, 0] >= 0.06415003]
V2 = curve2[:, 0][curve2[:, 0] >= 0.06415003]

f2 = scipy.interpolate.interp1d(P2, V2)

alpha = 4/5.
beta = 7/5.
def F(P, V): return -V**(alpha)*P - V**(beta) + V

def f(P):
    if -8.16 <= P <= 0.3849:
        return f2(P)
    elif -2119 <= P <= -8.837:
        return f1(P)
    else:
        roots = numpy.roots([-1, 0, 1, -P, 0, 0, 0, 0])
        return roots[numpy.nonzero(numpy.logical_and(numpy.isreal(roots), roots != 0))].real**5

f_vec = numpy.vectorize(f)

#line = numpy.linalg.lstsq(numpy.vstack([numpy.ones([1, len(P2[2000:])]), P2[2000:]]).T, V[2000:])[0]

#interpolant = numpy.vectorize(lambda p: f(p) if p > P[2000] else line[1]*p + line[0])

import pickle

pickle.dump(P, open('P.p', 'wb'))
pickle.dump(V, open('V.p', 'wb'))
pickle.dump(P2, open('P2.p', 'wb'))
pickle.dump(V2, open('V2.p', 'wb'))

#pickle.dump(f, open('interpolant.p', 'wb'))
