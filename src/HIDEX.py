from HIDEX_elements import Field, Kernel, inner, outer, CovarianceFunction
from pyLDS import LDS
import numpy as np

import sys, logging
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


# this class implements a HIDEX(p,q) spatiotemporal model. This represents
# multiple order dynamics, input and observation processes. Note that pretty
# much ALL of the computation is taken up by the "inner" methods of the basis
# functions which are described in the HIDEX_elements.py file.

class HIDEX:
	def __init__(self,F,G,H,f0,Q,R):
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
		"""
		# process and store arguments
		self.p = len(F)
		self.q = len(G)
		self.psi = Field(F[0].bases)
		self.phi = Field(f0.bases)
		self.F = F
		self.G = G
		self.H = H
		# form Phi = \int phi(s) phi^T(s) ds
		self.Phi = outer(self.phi,self.phi)
		# store the inversion
		self.Phi_inv = np.linalg.inv(self.Phi)
		# logging
		self.log = logging.getLogger('Field')
		self.log.info('Initialised new HIDEX model')
	
	def simulate(self,U):
		"""
		Parameters
		==========
		U : list
			list of input fields
		"""
		self.log.info('simulating HIDEX model')
		raise NotImplementedError
	
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
				self.Phi_inv * outer(self.phi, self.phi, G) for G in self.G
			])
			# this is [B_1 B_2 .. B_q; I 0]
			I = np.eye(len(self.phi)*(self.q-1))
			O = np.zeros((len(self.phi)*(self.q-1), len(self.phi)))
			B = np.vstack([B,np.hstack([I, O])])
		else: 
			B = None
		C = inner(self.H, self.phi)
		# make the covariance matrices
		Sw = inner(self.phi, self.phi, self.Q)
		# make a state space model
		ssmodel = LDS.LDS(A, B, C, Sw, self.R, x0)
		
	
	def estimate_fields(self,U,Y):
		"""
		Parameters
		==========
		U : list
			list of input fields
		Y : list
			list of observation vectors
		"""
		self.log.info('estimating fields')
		# form state space model
		ssmodel = self.gen_LDS()
		# use the smoother to extract the estimated field weights
		X,P,K,M = ssmodel.smoother(Y,[u.weights for u in U])
		# form the list of fields
		f = [Field(bases=self.phi,weights=x) for x in X]	
		# return the fields, the covariances and the cross covariances
		return f, P, M
	
	def estimate_kernels(f,g,Y):
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
		psi = Kernel(bases=self.psi,weights=ones(length(self.psi)))
		# then define an operator P_op based on that kernel
		P_op = lambda f: inner(psi,f)
		# initialise gammas
		gamma_a = np.empty(self.p)
		gamma_b = np.empty(self.q)
		Gamma_a = np.empty((self.p,self.p))
		Gamma_b = np.empty((self.q,self.q))
		Gamma_ab = np.empty((self.p,self.q))
		# form gammas
		for i in range(self.p):
			gamma_a[i] = inner(f[t], P_op(f[t-i]), Qinv)
		for j in range(self.q):
			gamma_b[j] = inner(f[t], P_op(f[t-i]), Qinv)
		gamma_c = inner(y[t], P_op(f[t]), Rinv)
		for i in range(self.p):
			for idash in range(self.p):
				Gamma_a[i,idash] = inner(P_op(f[t-i]), P_op(f[t-idash]), Qinv)
		for j in range(self.q):
			for jdash in range(self.q):
				Gamma_b[j,jdash] = inner(P_op(g[t-j]), P_op(g[t-jdash]), Qinv)
		for i in range(self.p):
			for j in range(self.q):
				Gamma_ab[i,j] = inner(P_op(f[t-i]), P_op(f[t-j]), Qinv)
		Gamma_c = inner(P_op(f[t]), P_op(f[t]), Rinv)
		# turn these into the large matrices
		ga = np.hstack(gamma_a)
		gb = np.hstack(gamma_b)
		# quick function to horizontally then vertically stack arrays
		form_block_matrix = lambda A: np.vstack([np.hstack(a) for a in A])
		Ga = form_block_matrix(Gamma_a)
		Gb = form_block_matrix(Gamma_b)
		Gab = form_block_matrix(Gamma_ab)
		# prepare some pretty syntax for the zeros matrix
		O = lambda m,n: zeros((m,n))
		na = self.p*len(self.psi)
		nb = self.q*len(self.psi)
		nc = len(gamma_c)
		# form the final large matrices
		gamma = np.hstack([ga,gb,gamma_c])
		Gamma = form_block_matrix([
			[Ga, Gab, O(na,nc)],
			[Gab.T, Gb, O(nb,nc)],
			[O(nc,na), O(nc,nb), Gamma_c]
		])	
	
	def estimate(Y):
		pass
	

if __name__ == "__main__":
	import os
	os.system("python paper_examples.py")


