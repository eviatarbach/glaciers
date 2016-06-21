from fractions import gcd
from functools import reduce

import autograd.numpy as numpy

def lcm(a, b):
    # Return the least common denominator of two non-negative numbers
    return a*b//gcd(a, b)

def roots_func(coeffs):
    coeffs = coeffs[::-1]
    coeffs = coeffs/coeffs[-1]
    n = len(coeffs) - 1
    rows = []
    rows.append([0]*(n - 1) + [-coeffs[0]])
    for i in range(1, n):
        rows.append([0]*(i - 1) + [1] + [0]*(n - i - 1) + [-coeffs[i]])
    return numpy.linalg.eigvals(numpy.array(rows))

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
        # poly = numpy.zeros(self.max_exponent + 1)
        # poly[self.new_exponents] = coeffs

        # Hack to work with autograd
        poly = [coeffs[numpy.where(self.new_exponents == j)[0]][0] if j in self.new_exponents else 0 for j in range(self.max_exponent + 1)]

        roots = roots_func(numpy.array(poly[::-1]))
        roots = numpy.real(roots[(numpy.imag(roots) == 0) & (numpy.real(roots) >= 0)])**self.recip_gcd

        #return numpy.unique(roots)
        return roots
