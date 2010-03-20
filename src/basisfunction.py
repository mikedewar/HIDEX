class BasisFunction:
	def __init__(self,dim):
		self.dim = dim
	
	def inner(self,other,weight):
		raise NotImplementedError

class Gaussian(BasisFunction):
	def __init__(self,dim):
		BasisFunction.__init__(self,dim)
	
	def inner(self,other,weight):
		# weight can be a Kernel or Covariance function  or None
		pass