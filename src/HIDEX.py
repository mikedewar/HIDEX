from HIDEX_elements import Field, Kernel, CovarianceFunction
from inner import inner, outer
from pyLDS import LDS
import numpy as np
import copy

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# this class implements a HIDEX(p, q) spatiotemporal model. This represents
# multiple order dynamics, input and observation processes. Note that pretty
# much ALL of the computation is taken up by the "inner" methods of the basis
# functions which are described in the HIDEX_elements.py file.

class HIDEX:
    
    def __init__(self, F, G, H, f0, Q, R, Pi0, obs_locns):
        """
        Parameters
        ==========
        F : list
            list of dynamic kernels
        G : list
            list of input kernels
        H : Kernel object
            observation kernel
        f0 : Field object
            initial field
        Q : CovarianceFunction object
            covariance of the field
        R : matrix
            covariance matrix of the observation noise
        Pi0 : matrix
            initial covariance matrix of the field weights
        obs_locns : list
            list of observation locations
        """
        # process and store arguments
        self.p = len(F)
        self.q = len(G)
        self.psi = F[0].bases
        self.phi = f0.bases
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.f0 = f0
        self.Pi0 = Pi0
        self.obs_locns = obs_locns
        # form Phi = \int phi(s) phi^T(s) ds
        self.Phi = outer(self.phi, self.phi)
        # store the inversion
        self.Phi_inv = np.linalg.inv(self.Phi)
        # logging
        self.log = logging.getLogger('Field')
        self.log.info('Initialised new HIDEX model')
        # genreate an LDS representation
        self.LDS = self.gen_LDS()

    def simulate(self, U):
        """
        Parameters
        ==========
        U : list
            list of input fields
        """
        self.log.info('simulating HIDEX model')
        # extract weights of input fields
        w = [np.matrix(u.weights).T for u in U]
        # simulate the underlying LDS
        X,Y = self.LDS.simulate(w)
        # form the hidden fields and return
        f_seq = np.empty(len(X),dtype=object)
        for i,x in enumerate(X):
            f = copy.copy(self.f0)
            f.weights = x
            f_seq[i] = f
        return f_seq,Y

    def gen_LDS(self):
        self.log.info('forming LDS model')
        # form the state space representation
        A = np.hstack([
            self.Phi_inv * outer(self.phi, self.phi, F) for F in self.F
        ])
        # this is [A_1 A_2 .. A_p; I 0]
        I = np.eye(len(self.phi)*(self.p-1))
        O = np.zeros((len(self.phi)*(self.p-1), len(self.phi)))
        A = np.vstack([A, np.hstack([I, O])])
        if self.G:
            B = np.hstack([
                self.Phi_inv * outer(self.phi, self.phi, G) 
                for G in self.G
            ])
            # this is [B_1 B_2 .. B_q; I 0]
            O = np.zeros((len(self.phi)*(self.p-1), self.q*len(self.phi)))
            B = np.vstack([B, O])
        else:
            B = None
        # C = <H,phi>(s1 ... sn)
        # <H,phi> = a^T <psi, phi>
        Cop = np.array([inner(self.H, phi_i) for phi_i in self.phi])
        # So we need to go through each basis function in Cop (C-operator) and
        # evalutate it at each observation location. This begs the question of
        # where the observation locations should come into the program. For 
        # the moment, though, we'll assume them fixed and store them in the 
        # HIDEX object
        C = np.zeros((len(self.obs_locns),len(self.phi)*(self.p)))
        for i,s in enumerate(self.obs_locns):
            for j,basis_function in enumerate(Cop):
                C[i,j] = basis_function(s)
        # make the covariance matrices
        Sw = np.zeros(
            (len(self.phi)*(self.p), len(self.phi)*(self.p))
            ,dtype=float
        )
        for i, basis_i in enumerate(self.phi):
            for j, basis_j in enumerate(self.phi):
                Sw[i,j] = inner(basis_i, basis_j, self.Q)
        # form the initial state
        x0 = np.hstack([self.f0.weights,np.zeros((len(self.phi)*(self.p-1)))])
        # turn everything into matrices for the LDS model
        A = np.matrix(A)
        B = np.matrix(B)
        C = np.matrix(C)
        Sw = np.matrix(Sw)
        Sv = np.matrix(self.R)
        x0 = np.matrix(x0).T
        Pi0 = np.matrix(self.Pi0)
        # make a state space model
        return LDS.LDS(A, B, C, Sw, Sv, x0, Pi0)


    def estimate_fields(self, U, Y):
        """
        Parameters
        ==========
        U : list
                list of input fields
        Y : list
                list of observation vectors
        """
        self.log.info('estimating fields')
        # use the smoother to extract the estimated field weights
        X, P, K, M = self.LDS.rtssmooth(Y, [np.matrix(u.weights).T for u in U])
        # form the list of fields
        f = [Field(bases=self.phi, weights=x) for x in X]
        # return the fields, the covariances and the cross covariances
        return f, P, M

    def estimate_kernels(self, f, g, Y):
        """
        Parameters
        ==========
        f : list
                list of hidden fields
        g : list
                list of input fields
        Y : array
                data
        """
        self.log.info('estimating kernels')
        # treat psi as a kernel with unit weights
        psi = Kernel(bases=self.psi, weights=np.ones(len(self.psi)))
        # then define an operator P_op based on that kernel
        P_op = lambda f: inner(psi, f)
        # initialise gammas
        gamma_a = np.empty(self.p)
        gamma_b = np.empty(self.q)
        Gamma_a = np.empty((self.p, self.p))
        Gamma_b = np.empty((self.q, self.q))
        Gamma_ab = np.empty((self.p, self.q))
        # form the time sequence
        time = range(len(f))
        # form gammas
        for t in time[1:]:
            for i in range(self.p):
                gamma_a[i] += inner(
                    f[t], 
                    P_op(f[t-i]), 
                    Qinv
                )
        
        for t in time[1:]:
            for j in range(self.q):
                gamma_b[j] += inner(
                    f[t], 
                    P_op(f[t-i]), 
                    Qinv
                )
        
        for t in time[self.p:]:
            gamma_c = inner(y[t], P_op(f[t]), Rinv)
            for i in range(self.p):
                for idash in range(self.p):
                    Gamma_a[i, idash] += inner(
                        P_op(f[t-i]), 
                        P_op(f[t-idash]), 
                        Qinv
                    )
            
        for t in time[self.q:]:
            for j in range(self.q):
                for jdash in range(self.q):
                    Gamma_b[j, jdash] += inner(
                        P_op(g[t-j]), 
                        P_op(g[t-jdash]), 
                        Qinv
                    )
            
        for t in time[max(self.q,self.p):]:
            for i in range(self.p):
                for j in range(self.q):
                    Gamma_ab[i, j] += inner(
                        P_op(f[t-i]), 
                        P_op(f[t-j]), 
                        Qinv
                    )
            
        for t in time:
            Gamma_c += inner(
                P_op(f[t]), 
                P_op(f[t]), 
                Rinv
            )
        
        # turn these into the large matrices
        ga = np.hstack(gamma_a)
        gb = np.hstack(gamma_b)
        # quick function to horizontally then vertically stack arrays
        form_block_matrix = lambda A: np.vstack([np.hstack(a) for a in A])
        Ga = form_block_matrix(Gamma_a)
        Gb = form_block_matrix(Gamma_b)
        Gab = form_block_matrix(Gamma_ab)
        # prepare some pretty syntax for the zeros matrix
        O = lambda m, n: zeros((m, n))
        na = self.p*len(self.psi)
        nb = self.q*len(self.psi)
        nc = len(gamma_c)
        # form the final large matrices
        gamma = np.hstack([ga, gb, gamma_c])
        Gamma = form_block_matrix([
            [Ga,        Gab,        O(na, nc)],
            [Gab.T,     Gb,         O(nb, nc)],
            [O(nc, na), O(nc, nb),  Gamma_c]
        ])
        # perform the maximisation
        theta = 0.5 * Gamma.I * gamma
        # unpick the parameters
        theta_l = array(theta).flatten().tolist()
        nx = len(self.psi)
        # watch carefully!
        # a is composed of the first p*nx elements of the list theta_l
        # so we pop these from the list. Then reshape them into an array
        # where each column of this array corresponds to a_i
        a = pb.array([
            theta_l.pop(0) for i in range(self.p*nx)
        ]).reshape(nx, self.p)
        # similarly, b is composed of the first q*nx of the now shortened 
        # list! so we just do the same thing, pop the first q*nx elements and 
        # reshape
        b = pb.array([
            theta_l.pop(0) for i in range(self.q*nx)
        ]).reshape(nx, self.q)
        # finally, c is all that's left. So just make it into an array!
        c = pb.array(theta_l)
        # form the new kernels. We need to transpose a and b because they're
        # stored columnwise, i.e. a_i = a[:, i] and the "for" works row-wise
        for Fi, ai in zip(self.F, a.T):
            Fi.bases = ai
        for Gi, bi in zip(self.G, b.T):
            Gi.bases = bi
        self.H.bases = c

    def estimate(Y, max_its=10, threshold = 0.001):
        not_converged = 1
        not_reached_max_its = 1
        i = 0
        # init
        f, P, M = self.estimate_fields(U, Y)
        while not_converged or not_reached_max_its:
            # M step
            self.estimate_kernels(f, g, Y, P, M)
            # E step
            f, P, M = self.estimate_fields(U, Y)
            # check for convergence
            i+=1
            if i == max_its:
                not_reached_max_its = 0
            try:
                if self.change_in_likelihood(Y) < threshold:
                    not_converged = 0
            except NotImplementedError:
                print "you should really do this at some point"

    def change_in_likelihood(self, Y):
        raise NotImplementedError


if __name__ == "__main__":
    import os
    os.system("python paper_examples.py")
