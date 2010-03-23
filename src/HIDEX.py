from HIDEX_elements import Field, Kernel, inner

class HIDEX:
	def __init__(self,F,G,H,f0,Q,R,psi,phi):
		"""
		Parameters
		==========
		F : list
			list of dynamic kernels
		G : list
			list of input kernels
		H : Kernel object
			observation kernel
		phi : list
			list of field basis functions
		psi : list
			list of kernel basis functions
		"""
			
		# TODO
		# ====
		# 1) There's probably not much need to include psi and phi here, when
		# they're going to be in the kernels and fields anyway. Still, for now,
		# it saves some time.

		self.p = len(F)
		self.q = len(G)
		self.psi = psi
		self.phi = phi
		self.F = F
		self.G = G
		self.H = H
		# form Phi = \int phi(s) phi^T(s) ds
		self.Phi = empty((len(self.phi),len(self.phi)))
		for i,phi_i in enumerate(self.phi):
			for phi_j in enumerate(self.phi):
				self.Phi[i,j] = inner(phi_i,phi_j)
		# store the inversion
		self.Phi_inv = np.inv(Phi)
	
	def simulate(self,U):
		"""
		Parameters
		==========
		U : list
			list of input fields
		"""
		pass
	
	def estimate_fields():
		# form the state space representation
		A = hstack([self.Phi_inv * inner(phi,phi,F) for F in self.F])
		B = hstack([self.Phi_inv * inner(phi,phi,G) for G in self.G])
		C 
		# this is [A_1 A_2 .. A_p; I 0]
		I = np.eye(len(self.phi)*(p-1))
		O = np.zeros((len(self.phi)*(p-1),len(self.phi)))
		A = vstack([A,hstack([I,O])])
		
		
	
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
