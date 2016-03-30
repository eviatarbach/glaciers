import numpy
from scipy.stats import gaussian_kde

def logit(x):
    return numpy.log(x/(1 - x))

class BoundedMultivariateKDE:
    def __init__(self, *args):
        self.data = []

        # Store components of g
        self.g = []

        # Store functions for determinant of Jacobian
        self.det = []

        for variable in args:
            data, bounds = variable
            b1, b2 = bounds
            self.data.append(data)
            if (b1 != None) and (b2 == None):
                # (a, oo)
                self.g.append(lambda x: numpy.log(x - b1))

                self.det.append(lambda y: numpy.exp(y))
            elif (b1 != None) and (b2 != None):
                # (a, b)
                self.g.append(lambda x: logit((x - b1)/(b2 - b1)))

                self.det.append(lambda y: (b2 - b1)*numpy.exp(y)/(numpy.exp(2*y) + 2*numpy.exp(y) + 1))
            elif (b1 == None) and (b2 == None):
                # (-oo, oo)
                self.g.append(lambda x: x)

                self.det.append(lambda y: 1)
            else:
                # (-oo, b) TODO
                raise NotImplementedError

        self.data = numpy.array(self.data)
        self.transformed_pdf = gaussian_kde(self.data).evaluate

    def pdf(self, point):
        y = [self.g[i](xi) for i, xi in enumerate(point)]
        diag_J = [self.det[i](yi) for i, yi in enumerate(y)]
        det = abs(numpy.prod(diag_J))
        return self.transformed_pdf(y)/det
