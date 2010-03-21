from kernel import Kernel
from field import Field
from basisfunction import BasisFunction
import numpy as np

def inner(A,B,C=None):
	if isinstance(A,Field) and isinstance(B,Field):
		return A.inner(B,C)
	if isinstance(A,Kernel) and isinstance(B,Field):
		# predicts the next field
		return A.inner(B,C)
	else:
		raise NotImplementedError