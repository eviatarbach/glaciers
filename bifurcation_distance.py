from data import equation, equation_diff, p, gamma

import numpy
import scipy.optimize


def system_G(G):
    def system(x):
        P, V = x
        F = equation.evaluate(numpy.array([1/4*G*P**2, -1/2*G*P, -P, 1/4*G, 1, -1]), V)
        dF = equation_diff.evaluate(numpy.array([1/4*G*P**2*(1/gamma - 1/p),
                                                 -1/4*G*(1/gamma + 1/p - 2),
                                                 -(1/gamma + 1/p), 1/2*G*P*(1/p - 1),
                                                 -P/gamma, 1]),
                                    V)
        return numpy.array([F, dF])
    return system


def P0(G):
    if G < -1:
        return numpy.inf
    elif G > 0:
        return scipy.optimize.root(system_G(G),
                                   numpy.array([numpy.log(G + 7)/5, G/25 if G > 2 else 0.1])).x[0]
    else:
        return scipy.optimize.root(system_G(G),
                                   numpy.array([numpy.log(G + 7)/5, 0.05*(G - 1.0) + 0.11])).x[0]

P0_vec = numpy.vectorize(P0)
