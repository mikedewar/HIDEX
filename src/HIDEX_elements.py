import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

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


class Kernel:
    def __init__(self,bases,weights):
        self.dim = bases[0].dim
        self.bases = bases
        self.weights = [weights]
        self.log = logging.getLogger('Kernel')
        self.log.debug("formed new Kernel")

    def inner(self,f):

        ### TODO

        if isinstance(f,Field):
            # this is \int K(s,s') f(s) ds
            # should return a field

            # the returned field's weights are the weight of the field

            # the basis functions of the field are the result of the inner
            # product of the kernel weights with the matrix-valued function
            # formed by the outer product of the kernel basis functions and
            # the field basis functions:
            #
            # = a^T \int psi(s,r) phi^T(r) dr x_t

            # so first form the outer product. We pretend psi and phi are
            # fields just so `outer` knows how to deal
            outer(Field(self.bases),Field(f.bases))

        elif isinstance(f,BasisFunction):
            self.log.debug("forming \int K(s,s') phi(s) ds")
            # this will yield a bunch of fields all with a single weight/basis
            fields = [b.inner(f) for b in self.bases]
            field_weights = [fi.weights[0] for fi in fields]
            try:
                weights = [w1*w2 for w1,w2 in zip(field_weights, self.weights)]
            except:
                print list(self.weights)
                print field_weights
                raise
            bases = [fi.bases[0] for fi in fields]
            # 
            assert all([isinstance(b,BasisFunction) for b in bases]), bases 
            try:
                return Field(bases,weights)
            except:
                self.log.debug("f: %s"%f)
                self.log.debug("b: %s"%self.bases[0])
                self.log.debug(self.bases[0].inner(f))
                raise
        else:
            raise NotImplementedError


class BasisFunction:
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('BasisFunction')

    def inner(self,other,weight):
        raise NotImplementedError


class Gaussian(BasisFunction):
    """
    Parameters
    ==========
    dim : scalar
            dimension of the basis function
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
        self.invwidth = width**-1
        self.constant = constant
        # consistency assertions
        if self.dim > 1:
            assert len(centre) == width.shape[0]
            assert len(centre) == width.shape[1]
        self.log.debug("formed new Gaussian BasisFunction")

    def inner(self,other):
        assert isinstance(other,Gaussian):
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
            # should return a Field
            self.log.debug("forming \int \psi(s,s') \phi(s) ds")
            # this is probably only going to work with ISOTROPIC
            # Qs of the form Q(s,s') = Q(s-s') centred at the
            # origin, and for 1D fields!!! Ugh! Really need to sit
            # down and do this with a coffee, some sun, and a nice
            # pen. Maybe a light breeze.
            
            # TODO this code is replicated in the Covariance
            # function code. Thought: Covariance is a special case
            # of a Kernel and/or a special case of a BasisFunction
            # so maybe it should inherit from one of these or
            # something
            
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
            basis = Gaussian(other.centre, width, constant)
            f = Field([basis],[constant])
            return f
        else:
            raise NotImplementedError
        
    


class CovarianceFunction(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.log = logging.getLogger('CovarainceFunction')
    

class SquaredExponential(CovarianceFunction):
    def __init__(self,dim,width,constant):
        """
        Parameters
        ==========
        dim : scalar
                dimension of the covariance function
        width : matrix
                width matrix of the covariance function
        constant :
                constant that premultiplies the covariance function
        """
        CovarianceFunction.__init__(self,dim)
        self.centre = np.zeros(dim)
        self.width = np.matrix(width)
        self.constant = constant
        self.invwidth = self.width.I
        self.log.debug("formed new SquaredExponential CovarianceFunction")
    

if __name__ == "__main__":
    import os
    os.system("python paper_examples.py")
