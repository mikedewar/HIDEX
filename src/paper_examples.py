import numpy as np
from HIDEX import *
from HIDEX_elements import *
import logging
import sys

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


# example 1
def example_1():
	# first order AR
	p = 1
	q = 0
	ny = 10
	# Kernels
	F = [Kernel([Gaussian((0,0),np.eye(2),1)],1)]
	G = []
	H = Kernel([Gaussian((0,0),np.eye(2),1)],1)
	# initial field
	f0 = Field([Gaussian(c,0.2,1) for c in range(10)])
	# field covaraince function
	Q = SquaredExponential((0,0),np.eye(2),1)
	# noise covariance matrix
	R = np.eye(ny)
	# HIDEX model
	model = HIDEX(F,G,H,f0,Q,R)
	# state space model representation
	ssmodel = model.gen_LDS()
	


if __name__ == "__main__":
	
	example_1()
