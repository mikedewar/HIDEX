import logging
import sys
import numpy as np
logging.basicConfig(stream=sys.stdout,level=logging.WARNING)

from HIDEX_elements import BasisFunction, Field, Kernel, CovarianceFunction

class Inner:
    def __init__(self, A, B, C):
        assert isinstance(A, (BasisFunction, Field, Kernel)), C
        assert isinstance(B, (BasisFunction, Field, Kernel)), B
        assert isinstance(C, (type(None), CovarianceFunction, Kernel)), C
        self.A = A
        self.B = B
        self.C = C
        self.log = logging.getLogger('Inner')
    
    def do_inner(self):
        
        if self.C is None:
            if isinstance(self.A, Kernel):
                if isinstance(self.B, Kernel):
                    raise NotImplementedError
                elif isinstance(self.B, Field):
                    out = self.kernel_field()
                    assert isinstance(out, Field)
                elif isinstance(self.B, BasisFunction):
                    out = self.kernel_basisfunction()
                    assert isinstance(out, Field)
        
            if isinstance(self.A, Field):
                if isinstance(self.B, Kernel):
                    out = self.field_kernel()
                    assert isinstance(out, Field)
                elif isinstance(self.B, Field):
                    out = self.field_field()
                    assert isinstance(out, float)   
                elif isinstance(self.B, BasisFunction):
                    out = self.field_basisfunction()
                    assert isinstance(out, float)
        
            if isinstance(self.A,  BasisFunction):
                if isinstance(self.B, Kernel):
                    out = self.basisfunction_kernel()
                    assert isinstance(out, Field)
                elif isinstance(self.B, Field):
                    out = self.basisfunction_field()
                    assert isinstance(out, float)
                elif isinstance(self.B, BasisFunction):
                    out = self.basisfunction_basisfunction()
                    if self.A.dim == 1 and self.B.dim == 1:
                        assert isinstance(out, float), out
                    elif self.A.dim == 2 and self.B.dim == 1:
                        assert isinstance(out, BasisFunction), out
                    else:
                        raise NotImplementedError
        
        else: # C is a covariance function or Kernel
            if isinstance(self.A, Field):
                if isinstance(self.B, Field):
                    out = self.field_covariance_field()
                    assert isinstance(out, float), out
            elif isinstance(self.A, BasisFunction):
                if isinstance(self.B, BasisFunction):
                    out = self.basisfunction_kernel_basisfunction()
                    assert isinstance(out, float), (type(out),out)
            else:
                print "\n"
                print isinstance(self.A, Field)
                print isinstance(self.B, Field)
                print self.C
                raise NotImplementedError
            
        
        return out

    def kernel_field(self):
        self.log.debug('forming <K,g>')
        # <K,g> = a^T <psi_K, phi_g^T> b
        # where a is the vector of weights of K and psi_K are the basis
        # functions of K. phi_g are the basis functions of g and b are g's 
        # weights
        Phi = outer(self.A.bases,self.B.bases) # <psi_K, phi_g^T>
        # we now need to flatten Phi and return a field with all the basis
        # functions in Phi as one long vector, and the appropriate product
        # of weights as the weights of the new field. Ugly!
        bases = Phi.flatten()
        weights = [ai*bj for ai in self.A.weights for bj in self.B.weights]
        return Field(bases=bases,weights=weights)
    
    def kernel_basisfunction(self):
        self.log.debug("forming <psi,f>")
        # <psi,f> = <psi,phi^T> b
        bases = np.empty(len(self.A.bases), dtype=object)
        for i, basis_i in enumerate(self.A.bases):
            bases[i] = inner(basis_i,self.B)
        return Field(bases=bases)

                
    def field_field(self):
        self.log.debug('forming <f,g>')
        # <f,g> = a^T <phi_f,phi_g> b
        # where a is the vector of weights of f and phi_f are the basis
        # functions
        Phi = np.empty(len(self.A.bases),len(self.B.bases))
        for basis_i in self.A.bases:
            for basis_j in self.B.bases:
                Phi[i,j] = inner(basis_i, basis_j)
        return np.dot(self.A.weights, np.dot(Phi, self.B.weights))
    
    def field_kernel(self):
        raise NotImplementedError
    
    def field_basisfunction(self):
        raise NotImplementedError
    
    def basisfunction_kernel(self):
        raise NotImplementedError
    
    def basisfunction_field(self):
        if self.A.dim == 1:
            self.log.debug("forming <phi,f>")
            # <phi,f> = <phi,phi^T> b
            Phi = np.empty(len(self.B.bases), dtype=float)
            for i,basis_i in enumerate(self.B.bases):
                Phi[i] = self.A.inner(basis_i)
            return np.dot(Phi,self.B.weights)
        elif self.A.dim == 2:
            self.log.debug("forming <psi,f>")
            # <psi,f> = <psi,phi^T> b
            bases = np.empty(len(self.B.bases), dtype=object)
            for i,basis_i in enumerate(self.B.bases):
                bases[i] = self.A.inner(basis_i)
            return hidex_el.Field(bases=bases, weights=self.B.weights)
        else:
            raise NotImplementedError
    
    def basisfunction_basisfunction(self):
        return self.A.inner(self.B)
    
    def field_covariance_field(self):
        self.log.debug("forming <f,g>_C")
        # <f,g>_C = a^T <phi_f,phi_g>_C b
        # <phi_f,phi_g>_C = \int\int phi_f(s) C(s,s') phi_g(s')  ds ds' 
        Phi = np.empty((len(self.A.bases),len(self.B.bases)))
        for i,basis_i in enumerate(self.A.bases):
            for j,basis_j in enumerate(self.B.bases):
                Phi[i,j] = inner(basis_i, inner(self.C, basis_j))
        return np.dot(self.A.weights, np.dot(Phi, self.B.weights))
    
    def basisfunction_kernel_basisfunction(self):
        self.log.debug("forming <phi,phi>_K")
        # <phi,phi>_K   = \int \int phi(s)K(s,s')phi(s') ds ds'
        #               = a^T <phi,<psi,phi>>
        bases = [inner(basis_i, self.B) for basis_i in self.C.bases]
        Phi = [inner(self.A,basis_i) for basis_i in bases]
        return np.dot(self.C.weights,Phi)
        
    def covariance_basisfunction(self):
        raise NotImplementedError
        
    

def inner(A, B, C=None):
    """
    Computes the inner produc of A and B, weighted by C
    """
    return Inner(A, B, C).do_inner()

def outer(A,B,C=None):
    """
    Computes a matrix <phi,phi^T> which is a pain. It's an outer product of 
    inner products.
    """
    if (A[0].dim == 1) and (B[0].dim==1):
        # will return a regular array of floats
        out = np.empty((len(A),len(B)),dtype=float)
    elif (A[0].dim == 2) and (B[0].dim == 1):
        # will return an array of basis functions
        out = np.empty((len(A),len(B)),dtype=object)
    else:
        print A[0].dim
        print B[0].dim
        raise NotImplementedError
    
    for i,phi_i in enumerate(A):
        for j,phi_j in enumerate(B):
            out[i,j] = inner(phi_i, phi_j, C)
    return out

if __name__ == "__main__":
    import HIDEX_elements as hidex_el
    import numpy as np
    K = hidex_el.Kernel([hidex_el.Gaussian((0,0),np.eye(2),1)],1)
    f = hidex_el.Field(
        [hidex_el.Gaussian(c,0.2,1) for c in range(10)],
        weights=None
    )
    g = hidex_el.Field(
        bases=[hidex_el.Gaussian(c,0.2,1) for c in range(10)],
        weights=np.random.randn(10)
    )
    b1 = hidex_el.Gaussian(0,1,1)
    b2 = hidex_el.Gaussian((0,0),np.eye(2),1)
    C = hidex_el.SquaredExponential(np.eye(2))
    
    print inner(K, b1)
    print inner(b1, b1)
    print inner(b2, b1)
    #
    print inner(f, f, C)
    print inner(f, g, C)
    print inner(f, g, K)
    
    
    

