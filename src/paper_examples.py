import numpy as np
from HIDEX import *
from HIDEX_elements import *
import logging
import sys
import hinton
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)


# example 1

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
Q = SquaredExponential(width=1*np.eye(2),const=0.1)
# noise covariance matrix
R = 0.1*np.eye(ny)
# state covariance
Pi0 = 100 * np.eye(len(f0.bases)*len(F))
# observation locations
obs_locns = np.linspace(0,10,ny)
# HIDEX model
model = HIDEX(F, G, H, f0, Q, R, Pi0, obs_locns)
# plot the output of the LDS
T = range(200)
g = []
for t in T:
    g.append(Field(
        bases = [Gaussian(float(c),0.2,1) for c in range(10)],
        weights = np.zeros(10)
    ))


for t in range(50,150):
    g[t].weights[4:7] = np.ones(3)


f,Y = model.simulate(g)
f_est,P,M = model.estimate_fields(g,Y)
f_est = [f0] + f_est

#model.estimate_kernels(f,g,Y)

if True:
    vmin = -2
    vmax = 5
    plt.figure()
    hinton.hinton(model.LDS.A)
    plt.figure()
    plt.subplot(1,3,1)
    Z = []
    for fi in f:
        Z.append([fi(s) for s in np.linspace(2,8,100)])
    Z_true = np.array(Z)
    plt.imshow(Z_true,vmin=vmin,vmax=vmax)
    plt.xlabel('space')
    plt.ylabel('time')
    plt.title('true field')
    plt.subplot(1,3,2)
    Z = []
    for fi in f_est:
        Z.append([fi(s) for s in np.linspace(2,8,100)])
    Z_est = np.array(Z)
    plt.imshow(Z_est,vmin=vmin,vmax=vmax)
    plt.xlabel('space')
    plt.ylabel('time')
    plt.title('estimated field')
    plt.subplot(1,3,3)
    plt.imshow(abs(Z_est-Z_true),vmin=vmin,vmax=vmax)
    plt.xlabel('space')
    plt.ylabel('time')
    plt.title('error')
    # plt.colorbar()
    plt.show()
