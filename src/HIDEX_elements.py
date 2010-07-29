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
    
    def plot(self,x,y):
        X,Y = np.meshgrid(x,y)
        z = np.zeros((len(x),len(y)))
        for basis in self.bases:
            for i in range(len(x)):
                for j in range(len(y)):
                    z[i,j] += basis((X[i,j],Y[i,j]))
        plt.imshow(z)

    
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
        if self.dim == 1:
            self.centre = centre
            self.width = width
            self.invwidth = width**-1
        else:
            self.centre = np.array(centre)
            self.width = np.array(width)
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
            cij=(self.width+other.width)/(self.width*other.width)
		    rij=(self.center-other.center)**2/(self.width+other.width)
		    constant = self.constant*other.constant*(np.pi)**(self.dim*0.5)
		    return constant*(cij**(-0.5*self.dim))*np.exp(-rij))
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
            assert len(x) == len(self.centre), (x,self.centre)
            return self.constant * np.exp(
                -0.5 * np.inner( np.inner(
                    (x - self.centre), self.invwidth),(x - self.centre))
            )
    
    def plot(self,x,*args):
        y = [self(xi) for xi in x]
        plt.plot(x,y,*args)

        
    
class CovarianceFunction(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('CovarainceFunction')
    

class SquaredExponential(CovarianceFunction):
    def __init__(self,width):
        """
        Parameters
        ==========
        width : scalar or matrix
                width of the covariance function
        constant :
                constant that premultiplies the covariance function
        """
        try:
            dim = width.shape[0]
        except AttributeError:
            dim = 1
        CovarianceFunction.__init__(self, dim)
        # we construct the covariance function as though it was a kernel, which
        # of course it is, except it's only ever going ot have one 'base' - a
        # Gaussian.
        if dim == 1:
            self.bases = [Gaussian(
                centre = 0,
                width = width,
                constant = 1
            )]
        else:
            self.bases = [Gaussian(
                centre = np.zeros(dim),
                width = np.matrix(width),
                constant = 1
            )]
        self.weights = [1]
        self.log.debug("formed new SquaredExponential CovarianceFunction")
    

if __name__ == "__main__":
    import os
    os.system("python paper_examples.py")
