import fractions

import numpy

from roots import RationalPowers


def closest_index_in_range(lower, upper, step, value):
    '''
    Find the index of the closest value to `value` in the range
    [lower, lower + step, ..., upper - step, upper] in constant time. `upper`
    must be greater than `lower`. If `value` is outside the range, return the
    corresponding boundary index (0 or the last index). When two values are
    equally close, the index of the smaller is returned.
    '''
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

RGI_REGIONS = ['Alaska', 'WesternCanadaUS', 'ArcticCanadaNorth',
               'ArcticCanadaSouth', 'GreenlandPeriphery', 'Iceland',
               'Svalbard', 'Scandinavia', 'RussianArctic', 'NorthAsia',
               'CentralEurope', 'CaucasusMiddleEast', 'CentralAsia',
               'SouthAsiaWest', 'SouthAsiaEast', 'LowLatitudes',
               'SouthernAndes', 'NewZealand', 'AntarcticSubantarctic']

RGI_NAMES = ['Alaska', 'Western Canada and USA', 'Arctic Canada (North)',
             'Arctic Canada (South)', 'Greenland (periphery)', 'Iceland',
             'Svalbard and Jan Mayen', 'Scandinavia', 'Russian Arctic',
             'North Asia', 'Central Europe', 'Caucasus and Middle East',
             'Central Asia', 'South Asia (West)', 'South Asia (East)',
             'Low Latitudes', 'Southern Andes', 'New Zealand',
             'Antarctic and Subantarctic']

p = fractions.Fraction(5, 3)
gamma = fractions.Fraction(5, 4)

# Equation is -Q*V^(7/5) + P*V^(2/5) - 2*P*V^(1/5) + P + V - V^(4/5)
equation = RationalPowers(numpy.array([fractions.Fraction(7, 5),
                                       fractions.Fraction(2, 5),
                                       fractions.Fraction(1, 5),
                                       0, 1,
                                       fractions.Fraction(4, 5)]))


def eq_volume(P, Q):
    return equation.find_root(numpy.array([-Q, P, -2*P, P, 1, -1]))
