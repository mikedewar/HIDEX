from kernel import Kernel
from field import Field
from numpy import np

class BasisFunction:
	def __init__(self,dim):
		self.dim = dim
	
	def inner(self,other,weight):
		raise NotImplementedError

class Gaussian(BasisFunction):
	def __init__(self,dim):
		BasisFunction.__init__(self,dim)
	
	def inner(self,other,weight):
		if isinstance(other,Field):
			if weight is None:
				# this is \int \phi(s) f(s) ds
				# should return a scalar
				return np.inner(
					[self.inner(basis) for basis in Field.bases],
					Field.weights
				)
			else:
				# this is \int \phi(s) Q(s,s') f(s')
				return inner(self.inner(weight),other)
		elif isinstance(other,BasisFunction):
			# this is \int \phi(s) \phi(s) ds
			# should return a scalar
			if isinstance(other,Guassian)
				# TODO actually need some calculation here
				pass
		elif isinstance(other,CovarianceFunction):
			# this is \int \phi(s) Q(s,s') ds
			# should return a BasisFunction
			if isinstance(other,SquareExponential):
				# TODO actually need some calculation here
				pass
		