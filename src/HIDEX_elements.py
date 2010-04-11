import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


class Field:
	def __init__(self,bases,weights=None):
		"""
		Parameters
		==========
		bases : list of `BasisFunction`s
			basis functions that decompose the field
		weights : list
			scalar weights of each basis function (default:1)
		"""
		self.dim = bases[0].dim
		self.bases = bases
		if weights is None:
			self.weights = np.ones(len(bases))
		else:
			self.weights = weights
		self.log = logging.getLogger('Field')
		self.log.debug("created new field")
	
	def __len__(self):
		return len(self.bases)
	
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
		elif type(other) == type(None):
			return self
		else:
			raise ValueError("trying to add %s to a Field!"%other)
	
	def inner(self,f,weight=None):
		if isinstance(f,Field):
			if weight is None:
				self.log.debug("forming \int f(s) g(s) ds")
				# should return a scalar
				Phi = np.empty((len(self.bases),len(f.bases)))
				for i,basis_i in enumerate(self.bases):
					for j,basis_j in enumerate(f.bases):
						Phi[i,j] = basis_i.inner(basis_j)
				# note this is a regular array inner product!
				return np.inner(np.inner(self.weights,Phi),f.weights)
			else:
				self.log.debug("forming \int f(s) Q(s,s') g(s') ds ds'")
				# should return a scalar
				return inner(inner(self,weight),f)
		elif isinstance(f,CovarianceFunction):
			self.log.debug("forming \int f(s) Q(s,s') ds")
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
		self.bases = bases
		self.weights = weights
		self.log = logging.getLogger('Kernel')
		self.log.debug("formed new Kernel")
	
	def inner(self,f):
		
		### TODO
		
		if isinstance(f,Field):
			# this is \int K(s,s') f(s) ds
			# should return a field
			# best way to do this is to form the ss representation, propagate
			# the field basis functions and then create a new field with the
			# propagated basis functions. 
			#
			# Not sure why this would ever be called on the Kernel!
			raise NotImplementedError
			
			
		elif isinstance(f,BasisFunction):
			self.log.debug("forming \int K(s,s') phi(s) ds")
			# should return a field	
			bases = [b.inner(f) for b in self.bases]
			try:
				return Field(bases,self.weights)
			except:
				self.log.debug("f: %s"%f)
				self.log.debug("b: %s"%self.bases[0])
				self.log.debug(self.bases[0].inner(f))
				raise
			
		else:
			raise NotImplementedError


class BasisFunction:
	def __init__(self,dim):
		self.dim = dim
		self.log = logging.getLogger('BasisFunction')
	
	def inner(self,other,weight):
		raise NotImplementedError
	

class Gaussian(BasisFunction):
	"""
	Parameters
	==========
	dim : scalar
		dimension of the basis function
	centre : scalar or matrix
		centre of the basis function
	width : scalar or matrix
		width of the basis function
	constant : scalar
		constant that premultiplies the basis function
	"""
	
	def __init__(self,centre,width,constant):
		try:
			dim = len(centre)
		except TypeError:
			dim = 1
		BasisFunction.__init__(self,dim)
		self.centre = centre
		self.width = width
		self.invwidth = width**-1
		self.constant = constant
		# consistency assertions
		if self.dim > 1:
			assert len(centre) == width.shape[0]
			assert len(centre) == width.shape[1]
		self.log.debug("formed new Gaussian BasisFunction")
	
	def inner(self,other,weight=None):
		if isinstance(other,Field):
			if weight is None:
				self.log.debug("forming \int \phi(s) f(s) ds")
				# should return a scalar
				return np.inner(
					[self.inner(basis) for basis in other.bases],
					other.weights
				)
			else:
				self.log.debug("forming \int \phi(s) Q(s,s') f(s') ds")
				return inner(self.inner(weight),other)
		elif isinstance(other,BasisFunction):
			if weight is None:
				if isinstance(other,Gaussian):
					# should return a scalar
					if self.dim == 1 and other.dim == 1:
						self.log.debug("forming \int \phi(s) \phi(s) ds")
						cij=self.width + other.width
						uij=cij**-1 * (self.width*self.centre + 
							other.width*other.centre)               
						rij=(self.centre**2 * self.width) + \
						       (other.centre**2 * other.width) - \
						       (uij**2 * cij)
						return self.constant * other.constant * (np.pi)**0.5 * \
						       (cij**-0.5) * np.exp(-rij)
					elif self.dim == 2 and other.dim == 1:
						self.log.debug("forming \int \psi(s,s') \phi(s) ds")
						# should return a Field
						
						# this is probably only going to work with ISOTROPIC Qs of the
						# form Q(s,s') = Q(s-s') centred at the origin, and for 1D 
						# fields!!! Ugh! Really need to sit down and do this with a
						# coffee, some sun, and a nice pen. Maybe a light breeze.
						
						# TODO this code is replicated in the Covariance
						# function code. Thought: Covariance is a special case
						# of a Kernel and/or a special case of a BasisFunction
						# so maybe it should inherit from one of these or something
						
						# this bit's crap
						if self.dim == 1:
							invsigma_Q = self.invwidth
						else:
							invsigma_Q = self.invwidth[0,0]
						# this bit needs work
						sum_invwidths = invsigma_Q + other.invwidth
						prod_invwidths = invsigma_Q * other.invwidth
						# this next line should be det(sum_invwidths)
						constant = np.pi**0.5 * sum_invwidths**0.5 
						width = sum_invwidths * (prod_invwidths)**-1
						basis = Gaussian(other.centre, width, constant)
						return Field([basis],[constant])
					else:
						self.log.debug("self.dim: %s"%self.dim)
						self.log.debug("other.dim: %s"%other.dim)
						raise NotImplementedError
				else:
					raise NotImplementedError
			else:
				self.log.debug("\int \phi(s) Q(s,s') \phi(s') ds ds'")
				# should return a scalar
				return self.inner(weight.inner(other))
		elif isinstance(other,CovarianceFunction):
			self.log.debug("fomring \int \phi(s) Q(s,s') ds")
			# should return a BasisFunction
			if isinstance(other,SquareExponential):
				# TODO actually need some calculation here
				raise NotImplementedError
			else:
				raise NotImplementedError
		else:
			raise NotImplementedErrorw
	

class CovarianceFunction:
	def __init__(self,dim):
		self.dim = dim
		self.log = logging.getLogger('CovarainceFunction')
	
	def inner(self,field):
		raise NotImplementedError
	
class SquaredExponential(CovarianceFunction):
	def __init__(self,dim,width,constant):
		"""
		Parameters
		==========
		dim : scalar
			dimension of the covariance function
		width : matrix
			width matrix of the covariance function
		constant : 
			constant that premultiplies the covariance function
		"""
		CovarianceFunction.__init__(self,dim)
		self.centre = np.zeros(dim)
		self.width = np.matrix(width)
		self.constant = constant
		self.invwidth = self.width.I
		self.log.debug("formed new SquaredExponential CovarianceFunction")
	
	def inner(self,f):
		if isinstance(f,Field):
			# this is \int Q(s,s') f(s) ds
			# should return a field
			return sum([self.inner(basis) for basis in f.bases])
			# each inner product above results in a field, and the sum of the
			# list of fields is the field composed of all the bases and 
			# weights of the component fields
		elif isinstance(f,BasisFunction):
			# this is \int Q(s,s') \phi(s) ds
			# should return a field with a single basis
			if isinstance(f,Gaussian):
				# this is probably only going to work with ISOTROPIC Qs of the
				# form Q(s,s') = Q(s-s') centred at the origin, and for 1D 
				# fields!!! Ugh! Really need to sit down and do this with a
				# coffee, some sun, and a nice pen. Maybe a light breeze.
				
				# this bit's crap
				invsigma_Q = self.invwidth[0,0]
				# this bit needs work
				sum_invwidths = invsigma_Q + f.invwidth
				prod_invwidths = invsigma_Q * f.invwidth
				# this next line should be det(sum_invwidths)
				constant = np.pi**0.5 * sum_invwidths**0.5 
				width = sum_invwidths * (prod_invwidths)**-1
				basis = Gaussian(f.dim, f.centre, width, constant)
				return Field([basis],[constant])
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

def inner(A,B,C=None):
	log = logging.getLogger('inner')
	if isinstance(A,Field) and isinstance(B,Field):
		return A.inner(B,C)
	elif isinstance(A,Kernel) and isinstance(B,Field):
		# predicts the next field
		return A.inner(B)
	elif isinstance(A,BasisFunction) and isinstance(B,BasisFunction):
		return A.inner(B,C)
	else:
		log.debug("problem routing the inner product <A,B>_C: ")
		log.debug("class of A: %s"%A.__class__)
		log.debug("class of B: %s"%B.__class__)
		log.debug("class of C: %s"%C.__class__)
		raise NotImplementedError


def outer(A, B, C = None):
	log = logging.getLogger('outer')
	if isinstance(A,Field) and isinstance(B,Field):
		out = np.empty((len(A.bases),len(B.bases)))
		for i, a in enumerate(A.bases):
			for j, b in enumerate(B.bases):
				out[i,j] = inner(a,b,C)
		return out
	else:
		log.debug("trying to form an invalid outer product")
		raise NotImplementedError
	


if __name__ == "__main__":
	import os
	os.system("python paper_examples.py")