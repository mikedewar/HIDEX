import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
logging.basicConfig(stream=sys.stdout,level=logging.WARNING)

class Field:
    def __init__(self,bases,weights=None):
        """
        Parameters
        ==========
        bases : list of `BasisFunction`s
                basis functions that decompose the field
        weights : list
                scalar weights of each basis function (default:1)
        """
        assert all([isinstance(b,BasisFunction) for b in bases]), bases
        self.dim = bases[0].dim
        self.bases = bases
        if weights is None:
            self.weights = np.ones(len(bases))
        else:
            self.weights = weights
        self.log = logging.getLogger('Field')
        self.log.debug("created new field")

    def __len__(self):
        return len(self.bases)

    def __add__(self,other):
        # the "sum" of two fields is formed by concatenating the bases and
        # weights of the two fields. Useful for combining the results of inner
        # products.
        if not isinstance(other,Field):
            raise ValueError("add only defined for two fields")
        return Field(
                bases = self.bases.append(other.bases),
                weights = self.weights.append(other.weights)
        )

    def __radd__(self,other):
        # this is defined just so the sum over a vector is defined
        if other == 0:
            return self
        elif type(other) == type(None):
            return self
        else:
            raise ValueError("trying to add %s to a Field!"%other)
    
    def __call__(self,x):
        return sum([weight*basis(x) 
            for weight,basis in zip(self.weights,self.bases)])


class Kernel:
    def __init__(self,bases,weights):
        self.dim = bases[0].dim
        self.bases = bases
        self.weights = weights
        self.log = logging.getLogger('Kernel')
        self.log.debug("formed new Kernel")
    
    def plot(self,x):
        y = np.zeros(len(x))
        for basis in self.bases:
            y += np.array([basis(xi) for xi in x])
        plt.plot(x,y)

    
class BasisFunction:
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('BasisFunction')

    def inner(self,other,weight):
        raise NotImplementedError
    
    def __call__(self,x):
        raise NotImplementedError


class Gaussian(BasisFunction):
    """
    Parameters
    ==========
    centre : scalar or matrix
            centre of the basis function
    width : scalar or matrix
            width of the basis function
    constant : scalar
            constant that premultiplies the basis function
    """
    def __init__(self,centre,width,constant):
        try:
            dim = len(centre)
        except TypeError:
            dim = 1
        BasisFunction.__init__(self,dim)
        self.centre = centre
        self.width = width
        if self.dim == 1:
            self.invwidth = width**-1
        else:
            self.invwidth = np.linalg.inv(width)
        self.constant = constant
        # consistency assertions
        if self.dim > 1:
            assert len(centre) == width.shape[0]
            assert len(centre) == width.shape[1]
        self.log.debug("formed new Gaussian BasisFunction")

    def inner(self,other):
        assert isinstance(other, (Gaussian,SquaredExponential))
        if self.dim == 1 and other.dim == 1:
            # should return a scalar
            self.log.debug("forming \int \phi(s) \phi(s) ds")
            cij=self.width + other.width
            uij=cij**-1 * (self.width*self.centre +
                    other.width*other.centre)
            rij=(self.centre**2 * self.width) + \
            (other.centre**2 * other.width) - \
                (uij**2 * cij)
            return self.constant * other.constant * (np.pi)**0.5 * \
                (cij**-0.5) * np.exp(-rij)
        elif self.dim == 2 and other.dim == 1:
            # should return a basis function
            self.log.debug("forming \int \psi(s,s') \phi(s) ds")
            # this is probably only going to work with ISOTROPIC
            # Qs of the form Q(s,s') = Q(s-s') centred at the
            # origin, and for 1D fields!!! Ugh! Really need to sit
            # down and do this with a coffee, some sun, and a nice
            # pen. Maybe a light breeze.
            
            # this bit's crap
            if self.dim == 1:
                invsigma_Q = self.invwidth
            else:
                invsigma_Q = self.invwidth[0,0]
            # this bit needs work
            sum_invwidths = invsigma_Q + other.invwidth
            prod_invwidths = invsigma_Q * other.invwidth
            # this next line should be det(sum_invwidths)
            constant = np.pi**0.5 * sum_invwidths**0.5
            width = sum_invwidths * (prod_invwidths)**-1
            return Gaussian(other.centre, width, constant)
        else:
            print self.dim
            print other.dim
            raise NotImplementedError
    
    def __call__(self,x):
        if self.dim == 1:
            return self.constant * np.exp(
                -0.5*(x-self.centre)**2 * self.invwidth
            )
        else:
            return self.constant * np.exp(
                -0.5 * np.inner( np.inner(
                    (x - self.centre), self.invwidth),(x - self.centre))
            )

        
    
class CovarianceFunction(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('CovarainceFunction')
    

class SquaredExponential(CovarianceFunction):
    def __init__(self,width):
        """
        Parameters
        ==========
        width : matrix
                width matrix of the covariance function
        constant :
                constant that premultiplies the covariance function
        """
        CovarianceFunction.__init__(self, width.shape[0])
        # we construct the covariance function as though it was a kernel, which
        # of course it is, except it's only ever going ot have one 'base' - a
        # Gaussian.
        self.bases = [Gaussian(
            centre = np.zeros(width.shape[0]),
            width = np.matrix(width),
            constant = 1
        )]
        self.weights = [1]
        self.log.debug("formed new SquaredExponential CovarianceFunction")
    

if __name__ == "__main__":
    import os
    os.system("python paper_examples.py")
