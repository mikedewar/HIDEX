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
    # number of observation locations
	ny = 10
	# Kernels
	F = [
	    Kernel(
	        bases = [
	            Gaussian((0.,0.), np.eye(2), 1),
	            Gaussian((0.,0.), 2*np.eye(2), 1),
	            Gaussian((0.,0.), 6*np.eye(2), 1)
	        ],
	        weights = np.array([10, -8, 5])*0.3
	    ),
	    Kernel(
	        bases = [
	            Gaussian((0.,0.), np.eye(2), 1),
	        ],
	        weights = np.array([-5])*0.2
	    )
	]
	G = [
	    Kernel(
	        bases = [Gaussian((0.,0.),np.eye(2),1)],
	        weights = [1]
	    ),
	]
	H = Kernel([Gaussian((0.,0.),np.eye(2),1)],1)
	# initial field
	f0 = Field([Gaussian(float(c),0.2,1) for c in range(10)])
	# field covaraince function
	Q = SquaredExponential(width=10*np.eye(2),const=0.1)
	# noise covariance matrix
	R = 0.1*np.eye(ny)
	# state covariance
	Pi0 = 100 * np.eye(len(f0.bases)*len(F))
	# observation locations
	obs_locns = np.linspace(0,10,ny)
	# HIDEX model 
	model = HIDEX(F, G, H, f0, Q, R, Pi0, obs_locns)
	plt.figure()
	hinton.hinton(model.LDS.A)
	# plot the output of the LDS
	T = range(100)
	U = [np.matrix(np.zeros(10)).T for t in T]
	for i in range(50,80):
	    U[i] = np.matrix(np.ones(10)).T
	X,Y = model.LDS.simulate(U)
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(np.hstack(X).T)
	plt.subplot(1,2,2)
	plt.plot(np.hstack(Y).T)
	plt.show()

if __name__ == "__main__":
	
	example_1()
