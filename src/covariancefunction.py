class CovarianceFunction:
	def __init__(self,dim):
		self.dim = dim
	
	def inner(self,field):
		raise NotImplementedError
	
class SquaredExponential(CovarianceFunction):
	def inner(self,f):
		if isinstance(f,Field):
			# this is \int Q(s,s') f(s) ds
			# should return a field
			return sum([self.inner(basis) for basis in f.bases])
			# each inner product above results in a field, and the sum of the
			# list of fields is the field composed of all the bases and 
			# weights of the component fields
		elif isinstance(f,BasisFunction):
			# this is \int Q(s,s) \phi(s) ds
			# should return a field with a single basis
			if isinstance(f,Gaussian):
				# TODO actually need some calculation here
				pass
			else:
				raise NotImplementedError 
			