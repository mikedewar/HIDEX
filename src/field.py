from basisfunction import BasisFunction
from covariancefunction import CovarianceFunction
from inner import inner
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
			bases = self.bases.append(other.bases)
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
				for i,basis_i in enumerate(self.bases)
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