from kernel import Kernel
from field import Field
from basisfunction import BasisFunction
import numpy as np

def inner(A,B,C=None):
	
	# this condition is int_s K f ds
	if isinstance(A,Kernel) and isinstance(B,Field):
		if C is None:
			return inner(
				inner(
					A.parameters, inner(A.bases,B.bases)
				),
				B.parameters
			)
		else:
			raise NotImplementedError
	
	# this condition is \int\int f Q g dsds'		
	elif isinstance(A,Field) and isinstance(B,Field):
		return inner(
			inner(
				A.parameters, inner(A.bases,B.bases,C)
			),
			B.parameters
		)
	
	# this condition is \int\int psi Q psi ds ds and is one of two
	# base calls in this function
	elif isinstance(A,BasisFunction) and isinstance(B,BasisFunction):
		return A.inner(B,C)
		
	# this condition is \int psi f ds
	elif isinstance(A,BasisFunction) and isinstance(B,Field):
		return inner(inner(A,B.bases,C),B.weights)
		
	elif isinstance(A,np.ndarray) and isinstance(B,np.ndarray):
		# this could either be two arrays of scalars, or it could be
		# an array of bases and an array of scalars.
		if A.dtype is np.dtype('object'):
			# bases times weights (returns a kernel or a field)
			return np.
		
		
		if C is None:
			return np.inner(A,B)
		else:
			return np.inner(np.inner(A,C),B)
	
	else:
		raise NotImplementedError 
		

