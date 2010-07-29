import numpy as np
from HIDEX import *
from HIDEX_elements import *
import logging
import sys
import hinton
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


# example 1
def example_1():
	# first order AR
	p = 1
	q = 0
	ny = 50
	# Kernels
	F = [
	    Kernel(
	        bases = [
	            Gaussian(0, 1, 1),
	            Gaussian(0, 1, 1),
	            Gaussian(0, 1, 1)
	        ],
	        weights = [0.1, -0.05]),
	]
	G = [
	    Kernel(
	        bases = [Gaussian(0,1,1)],
	        weights = [0.1]
	    ),
	]
	H = Kernel([Gaussian(0,1,1)],1)
	# initial field
	f0 = Field([Gaussian(c,0.2,1) for c in range(10)])
	# field covaraince function
	Q = SquaredExponential(width=1)
	# noise covariance matrix
	R = np.eye(ny)
	# state covariance
	Pi0 = 100 * np.eye(len(f0.bases)*p)
	# observation locations
	obs_locns = np.linspace(0,10,ny)
	# HIDEX model 
	model = HIDEX(F, G, H, f0, Q, R, Pi0, obs_locns)
	plt.figure()
	hinton.hinton(model.LDS.A)
	plt.figure()
	model.F[0].plot(np.linspace(-5,5,100))
	plt.show()
	


if __name__ == "__main__":
	
	example_1()
