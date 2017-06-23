import fractions

import numpy

from roots import RationalPowers


def closest_index_in_range(lower, upper, step, value):
    """
    Find the index of the closest value to `value` in the range
    [lower, lower + step, ..., upper - step, upper] in constant time.
    `upper` must be greater than `lower`.  If `value` is outside the
    range, return the corresponding boundary index (0 or the last
    index).  When two values are equally close, the index of the smaller
    is returned.
    """
    if value >= upper:
        return int((upper - lower)/step)
    elif value < lower:
        return 0

    value = value - lower
    upper = upper - lower
    lower = 0

    index = int(value//step + 1)

    if step*index - value >= value - step*(index - 1):
        index -= 1

    return index


RGI_REGIONS = ['Alaska', 'WesternCanadaUS', 'ArcticCanadaNorth', 'ArcticCanadaSouth',
               'GreenlandPeriphery', 'Iceland', 'Svalbard', 'Scandinavia', 'RussianArctic',
               'NorthAsia', 'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia', 'SouthAsiaWest',
               'SouthAsiaEast', 'LowLatitudes', 'SouthernAndes', 'NewZealand',
               'AntarcticSubantarctic']

RGI_NAMES = ['Alaska', 'Western Canada and USA', 'Arctic Canada (North)', 'Arctic Canada (South)',
             'Greenland (periphery)', 'Iceland', 'Svalbard and Jan Mayen', 'Scandinavia',
             'Russian Arctic', 'North Asia', 'Central Europe', 'Caucasus and Middle East',
             'Central Asia', 'South Asia (West)', 'South Asia (East)', 'Low Latitudes',
             'Southern Andes', 'New Zealand', 'Antarctic and Subantarctic']

THICK_REGIONS = ['alaska', 'westerncanada', 'arcticcanadaN', 'arcticcanadaS', 'greenland',
                 'iceland', 'svalbard', 'scandinavia', 'russianarctic', 'northasia',
                 'centraleurope', 'caucasus', 'centralasiaN', 'centralasiaW', 'centralasiaS',
                 'lowlatitudes', 'southernandes', 'newzealand', 'antarctic']

# The amount to adjust the median, area-weighted mean, or mid-range altitude to estimate the
# balanced-budget ELA
ELA_CONV = {'ela_mid': -27,  # Braithwaite & Raper, 2009
            'ela_weighted': -36}  # Braithwaite, 2015

ERRS = {'height': 0.3,  # estimated relative error in height (Huss & Farinotti, 2012)
        'length': 0.2,  # estimated relative error in length (Machguth & Huss, 2014)
        'vol_interp': 0.223,  # root-mean square relative error in interpolating volume
        'length_interp': 0.249,  # root-mean square relative error in interpolating length
        'g_abl': 0.003735,  # root-mean square error in interpolating g_abl
        'g_acc': 0.001958,  # root-mean square error in interpolating g_acc
        'ela_mid': 125,  # standard deviation of error distribution (Braithwaite & Raper, 2009)
        'ela_weighted': 56}  # standard deviation of error distribution (Braithwaite, 2015)

gamma = fractions.Fraction(5, 4)
p = gamma + (gamma*(1 - gamma))/(gamma - 2)

# Equation is 1/4*G*P^2*V^(1/gamma - 1/p) - 1/2*G*P*V^(-1/p + 1) - P*V^(1/gamma)
#              + 1/4*G*V^(-1/gamma - 1/p + 2) + V - V^(1/gamma + 1/p)

equation = RationalPowers(numpy.array([1/gamma - 1/p, 1 - 1/p, 1/gamma, 2 - 1/gamma - 1/p,
                                       1, 1/gamma + 1/p]))

# Derivative of the above, in order to evaluate stability
# 1/4*G*P^2*V^(1/gamma - 1/p - 1)*(1/gamma - 1/p)
#  - 1/4*G*V^(-1/gamma - 1/p + 1)*(1/gamma + 1/p - 2)
#  - V^(1/gamma + 1/p - 1)*(1/gamma + 1/p) + 1/2*G*P*(1/p - 1)/V^(1/p)
#  - P*V^(1/gamma - 1)/gamma + 1

equation_diff = RationalPowers(numpy.array([1/gamma - 1/p - 1, -1/gamma - 1/p + 1,
                                            1/gamma + 1/p - 1, -1/p, 1/gamma - 1, 0]))


def eq_volume(G, P):
    return equation.find_roots(numpy.array([1/4*G*P**2, -1/2*G*P, -P, 1/4*G, 1, -1]))


def final_volume(G, P, V):
    eq_volumes = eq_volume(G, P)
    loc = eq_volumes.searchsorted(V)
    if loc == len(eq_volumes):
        # If the current volume is larger than the largest equilibrium
        # volume, return the largest since the volume must be finite
        # and non-negative
        return eq_volumes[loc - 1]
    lower, upper = (loc - 1, loc) if loc != 0 else (loc, loc + 1)
    if stability(G, P, eq_volumes[lower]) == -1:
        return eq_volumes[lower]
    else:
        return eq_volumes[upper]


def stability(G, P, V):
    terms = numpy.array([1/4*G*P**2*(1/gamma - 1/p), -1/4*G*(1/gamma + 1/p - 2),
                         -(1/gamma + 1/p), 1/2*G*P*(1/p - 1), -P/gamma, 1])
    if V == 0:
        # As V -> 0, the term with the most negative exponent dominates
        nonzero = numpy.nonzero(terms)
        return numpy.sign(terms[nonzero][numpy.argmin(equation_diff.exponents[nonzero])])
    else:
        return numpy.sign(equation_diff.evaluate(terms, V))


def diff(G, P, V, dP=1e-8):
    return numpy.gradient([final_volume(G, P - dP, V), final_volume(G, P, V),
                           final_volume(G, P + dP, V)], dP)[1]


final_volume_vec = numpy.vectorize(final_volume)
diff_vec = numpy.vectorize(diff)

gamma = float(gamma)
p = float(p)
