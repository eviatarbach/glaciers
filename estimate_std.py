import numpy
import scipy.optimize


def obj_ar(xi, resid, mu):
    eta, theta = xi
    return sum((resid - eta*mu**theta)**2)


def grad_ar(xi, resid, mu):
    eta, theta = xi
    grad1 = sum(-mu**theta*2*(resid - eta*mu**theta))
    grad2 = sum(2*(eta*mu**theta - resid)*eta*mu**theta*numpy.log(mu))
    return numpy.array([grad1, grad2])


def hess_ar(xi, resid, mu):
    eta, theta = xi
    return numpy.array([[sum(2*mu**(2*theta)),
                         sum(2*(2*eta*mu**(2*theta) - abs(y - mu)*mu**theta)*numpy.log(mu))],
                        [sum(2*(2*eta*mu**(2*theta) - abs(y - mu)*mu**theta)*numpy.log(mu)),
                         sum(2*(2*eta**2*mu**(2*theta) - eta*abs(y - mu)*mu**theta)*numpy.log(mu)**2)]])


def obj_weighted(xi, resid, mu, theta_ar):
    eta, theta = xi
    return sum((resid - eta*mu**theta)**2/(mu**(2*theta_ar)))


def grad_weighted(xi, resid, mu, theta_ar):
    eta, theta = xi
    grad1 = sum(2*(eta*mu**theta - resid)*mu**theta/mu**(2*theta_ar))
    grad2 = sum(2*(eta*mu**theta - resid)*eta*mu**theta*numpy.log(mu)/mu**(2*theta_ar))
    return numpy.array([grad1, grad2])


def estimate_std_function(y, mu, eta0, theta0, theta_bounds=(-10, 10)):
    """
    Given a set of points (x, y), as well as a mean mu evaluated at
    each x and initial values eta0 and theta0, estimate the standard
    deviation function sigma(mu) = eta*mu^theta using L-BFGS-B.

    Optionally, provide bounds on theta (default -10 <= theta <= 10).
    If the upper bound on theta is too high, the optimization is prone
    to experience overflow.

    Method from M. Davidian and R. J. Carroll, "Variance Function
    Estimation", Journal of the American Statistical Association,
    Vol. 82, No. 400 (Dec., 1987), pp. 1079--1091.
    """

    # We are actually estimating the mean absolute deviation, which is related to the standard
    # deviation by a factor of sqrt(2/pi)
    eta0 = eta0*numpy.sqrt(2/numpy.pi)

    resid = abs(y - mu)

    res = scipy.optimize.fmin_l_bfgs_b(lambda xi: obj_ar(xi, resid, mu), [eta0, theta0],
                                       fprime=lambda xi: grad_ar(xi, resid, mu),
                                       bounds=[(0, None), theta_bounds], factr=1)
    res2 = scipy.optimize.fmin_l_bfgs_b(lambda xi: obj_weighted(xi, resid, mu, res[0][1]), res[0],
                                        fprime=lambda xi: grad_weighted(xi, resid, mu, res[0][1]),
                                        bounds=[(0, None), theta_bounds], factr=1)

    eta, theta = res2[0]

    return [eta/numpy.sqrt(2/numpy.pi), theta], res2
