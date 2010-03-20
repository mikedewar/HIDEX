from inner import inner

class HIDEX:
	def __init__(self,F,G,H,f0,Q,R,psi,phi):	
		self.p
		self.q
		self.psi = psi
		self.phi = phi
	
	def simulate(self,T,U):
		pass
	
	def estimate_fields():
		pass
	
	def estimate_kernels(f,u_seq,Y):
		"""
		Parameters
		==========
		f : list
			list of hidden fields
		u_seq : list
			list of input fields
		Y : array
			data
		"""
		P_op = lambda f: inner(self.psi,f)
		
		gamma_a = [inner(f[t], P_op(f[t-i])), Qinv) for i in range(self.p)]
		gamma_b = [inner(f[t], P_op(f[t-i])), Qinv) for j in range(self.q)]
		gamma_c = inner(y[t], P_op(y[t]), Rinv)
		for i in range(self.p):
			for idash in range(self.p):
				Gamma_a[i,idash] = inner(P_op(f[t-i]), P_op(f[t-idash]), Qinv)
		for j in range(self.q):
			for jdash in range(self.q):
				Gamma_b[j,jdash] = inner(P_op(g[t-j]), P_op(g[t-jdash]), Qinv)
		for i in range(self.p):
			for j in range(self.q):
				Gamma_ab[i,j] = inner(P_op(f[t-i]), P_op(f[t-j]), Qinv)
		Gamma_c = inner(P_op(y[t]), P_op(y[t]), Rinv)

	
	def estimate(Y):
		pass


