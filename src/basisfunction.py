class BasisFunction:
	def __init__(self):
		pass
	
	def inner(self,other):
		raise NotImplementedError
	
	def convolve(self,other):
		raise NotImplementedError

class Gaussian(BasisFunction):
	def __init__(self):
		pass