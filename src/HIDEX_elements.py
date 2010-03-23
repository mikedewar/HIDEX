import numpy as np

class Field:
	def __init__(self,bases,weights=None):
		self.dim = bases[0].dim
		self.bases = bases
		if weights is None:
			self.weights = np.ones(len(bases))
		else:
			self.weights = weights
	
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
		if other == 0:
			return self
		else:
			raise ValueError("trying to add %s to a Field!"%other)
	
	def inner(self,f,weight=None):
		if isinstance(f,Field):
			if weight is None:
				# this is \int f(s) f(s) ds
				# should return a scalar
				Phi = np.empty((len(self.bases),len(f.bases)))
				for i,basis_i in enumerate(self.bases):
					for j,basis_j in enumerate(f.bases):
						Phi[i,j] = basis_i.inner(basis_j)
				# note this is a regular array inner product!
				return np.inner(np.inner(self.weights,Phi),f.weights)
			else:
				# this is \int f(s) Q(s,s') g(s') ds ds'
				# should return a scalar
				return inner(inner(self,weight),f)
		if isinstance(f,CovarianceFunction):
			# this is \int f(s) Q(s,s') ds
			# should return a field
			return Field(
				weights = self.weights,
				bases = np.array([basis.inner(f) for basis in self.bases])
			)
		else:
			raise NotImplementedError
		
	

class Kernel:
	def __init__(self,bases,weights):
		self.dim = bases[0].dim
	
	def inner(self,field):
		pass

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
			if isinstance(other,Guassian):
				# TODO actually need some calculation here
				pass
			else:
				raise NotImplementedError
		elif isinstance(other,CovarianceFunction):
			# this is \int \phi(s) Q(s,s') ds
			# should return a BasisFunction
			if isinstance(other,SquareExponential):
				# TODO actually need some calculation here
				pass
			else:
				raise NotImplementedError
		else:
			raise NotImplementedErrorw
	

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
		else:
			raise NotImplementedError

def inner(A,B,C=None):
	if isinstance(A,Field) and isinstance(B,Field):
		return A.inner(B,C)
	if isinstance(A,Kernel) and isinstance(B,Field):
		# predicts the next field
		return A.inner(B,C)
	else:
		raise NotImplementedError			