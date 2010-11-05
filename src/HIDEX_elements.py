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
        return np.sum([weight*basis(x) 
            for weight,basis in zip(self.weights,self.bases)])
    
    def plot(self,s):
        plt.plot(s,[self(si) for si in s])


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
    const : scalar
            const that premultiplies the basis function
    """
    def __init__(self,centre,width,const):
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
        self.const = const
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
            rij=(self.centre-other.centre)**2/(self.width+other.width)
            const = self.const*other.const*(np.pi)**(self.dim*0.5)
            return const*(cij**(-0.5*self.dim))*np.exp(-rij)
        elif self.dim == 2 and other.dim == 1:
            # should return a basis function
            # will only work with zero-mean isotropic kernels
            assert all(self.centre.flatten() == np.array([0,0])), self.centre
            assert all(np.diag(self.width) == self.width[0,0]), self.width
            a = self.width[0,0]
            self.log.debug("forming \int \psi(s,s') \phi(s) ds")
            const = (self.const*other.const)*(np.pi*a*other.width)/(a+other.width)
            width = a + other.width
            centre = float(other.centre)
            assert isinstance(centre,float), centre
            assert isinstance(width,float), width
            assert isinstance(const,float), const
            return Gaussian(centre,width,const)
        else:
            print self.dim
            print other.dim
            raise NotImplementedError
    
    def __call__(self,x):
        d = x - self.centre
        if self.dim == 1:
            return self.const * np.exp(-0.5*d**2 * self.invwidth)
        else:
            assert len(x) == len(self.centre), (x,self.centre)
            return self.const * np.exp(
                -0.5 * np.dot(np.dot(d,self.invwidth),d)
            )
    
    def plot(self,x,*args):
        y = [self(xi) for xi in x]
        plt.plot(x,y,*args)

        
    
class CovarianceFunction(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('CovarainceFunction')
    

class SquaredExponential(CovarianceFunction):
    def __init__(self,width,const=1):
        """
        Parameters
        ==========
        width : scalar or matrix
                width of the covariance function
        const :
                const that premultiplies the covariance function
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
                const = const
            )]
        else:
            self.bases = [Gaussian(
                centre = np.zeros(dim),
                width = np.matrix(width),
                const = const
            )]
        self.weights = [1]
        self.log.debug("formed new SquaredExponential CovarianceFunction")
    
    def __call__(self,si,sj):
        return self.bases[0]([si,sj])
    

if __name__ == "__main__":
    import os
    os.system("python paper_examples.py")
