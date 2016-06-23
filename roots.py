from fractions import gcd
from functools import reduce

import numpy


def lcm(a, b):
    """Return the least common denominator of two non-negative numbers."""
    return a*b//gcd(a, b)


class RationalPowers:
    def __init__(self, exponents):
        """
        Put all the exponent logic in here so once we have the
        coefficients we can more quickly find the roots.
        """
        denom_lcm = reduce(lcm, [exponent.denominator for exponent in exponents])
        num_gcd = reduce(gcd, [exponent.numerator for exponent in exponents])
        recip_gcd = denom_lcm/num_gcd
        new_exponents = (exponents*recip_gcd).astype(int)

        self.max_exponent = max(new_exponents)
        self.new_exponents = new_exponents
        self.exponents = exponents
        self.recip_gcd = recip_gcd

    def evaluate(self, coeffs, x):
        return sum(coeffs*x**self.exponents)

    def find_root(self, coeffs):
        poly = numpy.zeros(self.max_exponent + 1)
        poly[self.new_exponents] = coeffs

        roots = numpy.roots(numpy.array(poly[::-1]))
        roots = numpy.real(roots[(numpy.imag(roots) == 0)
                                 & (numpy.real(roots) >= 0)])**self.recip_gcd

        return numpy.unique(roots)
