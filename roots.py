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

        Examples:
            Initialize a function with exponents 5/3 and 7/2

            >>> RationalPowers([fractions.Fraction(5, 3), fractions.Fraction(7, 2)])
        """
        exponents = numpy.array(exponents)
        denom_lcm = reduce(lcm, [exponent.denominator for exponent in exponents])
        num_gcd = reduce(gcd, [exponent.numerator for exponent in exponents])
        recip_gcd = denom_lcm/num_gcd
        new_exponents = (exponents*recip_gcd).astype(int)

        self.max_exponent = max(new_exponents)
        self.new_exponents = new_exponents
        self.exponents = exponents
        self.recip_gcd = recip_gcd

    def evaluate(self, coeffs, x):
        """
        Evaluate the function with coefficients `coeffs` at `x`.

        Examples:
            Evaluate the function f(x) = x**(5/3) + 3*x**(7/2) at x = 3

            >>> function = RationalPowers([fractions.Fraction(5, 3), fractions.Fraction(7, 2)])
            >>> function.evaluate([1, 3])
            146.53636688223477
        """
        return sum(coeffs*x**self.exponents)

    def find_roots(self, coeffs):
        """
        Return the real, unique roots of the function with coefficients `coeffs`.

        Examples:
            Find the roots of the function f(x) = 0.6*x**(5/3) + 0.1*x - 10*x**(3/4) + 5*x**(1/7)

            >>> function = RationalPowers([fractions.Fraction(5, 3), fractions.Fraction(1, 1),
                                           fractions.Fraction(3, 4), fractions.Fraction(1, 7)])
            >>> roots = function.find_roots([0.6, 0.1, -10, 5])
            >>> roots
            array([  0.        ,   0.33552363,  19.08603779])

            Verify that these are indeed roots:
            
            >>> [function.evaluate([0.6, 0.1, -10, 5], root) for root in roots]
            [0.0, -1.3677947663381929e-13, 1.7498003046512167e-11]
        """
        poly = numpy.zeros(self.max_exponent + 1)
        poly[self.new_exponents] = coeffs

        roots = numpy.roots(numpy.array(poly[::-1]))
        roots = numpy.real(roots[(numpy.imag(roots) == 0)
                                 & (numpy.real(roots) >= 0)])**self.recip_gcd

        return numpy.unique(roots)
