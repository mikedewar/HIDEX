import numpy as np
from HIDEX import *

## example 1
p = 2 # autoregressive order
q = 1 # input order
# initialise kernels
F = np.empty(p, Kernel)
G = np.empty(q, Kernel)
# initialise fields
f = Field()
g = Field()