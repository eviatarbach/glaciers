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

p = fractions.Fraction(5, 3)
gamma = fractions.Fraction(5, 4)

# Equation is -Q*V^(4/5) - V^(7/5) + P*V^(1/5) + V - 2*P*V^(2/5)/Q + P*V^(3/5)/Q^2
equation = RationalPowers(numpy.array([fractions.Fraction(4, 5), fractions.Fraction(7, 5),
                                       fractions.Fraction(1, 5), 1, fractions.Fraction(2, 5),
                                       fractions.Fraction(3, 5)]))

# Derivative of the above, in order to evaluate stability
# -4/5*Q/V^(1/5) - 7/5*V^(2/5) + 1/5*P/V^(4/5) - 4/5*P/(Q*V^(3/5)) + 3/5*P/(Q^2*V^(2/5)) + 1
equation_diff = RationalPowers(numpy.array([fractions.Fraction(-1, 5), fractions.Fraction(2, 5),
                                            fractions.Fraction(-4, 5), fractions.Fraction(-3, 5),
                                            fractions.Fraction(-2, 5), 0]))


def eq_volume(P, Q):
    return equation.find_root(numpy.array([-Q, -1, P, 1, -2*P/Q, P/Q**2]))


def stability(P, Q, V):
    terms = numpy.array([-4/5*Q, -7/5, 1/5*P, -4/5*P/Q, 3/5*P/Q**2, 1])
    if V == 0:
        # As V -> 0, the term with the most negative exponent dominates
        return numpy.sign(terms[2])
    else:
        return numpy.sign(equation_diff.evaluate(terms, V))
