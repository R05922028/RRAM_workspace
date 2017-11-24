import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import simps
import numpy as np
import matplotlib.mlab as mlab
import math
import sys
import mpl_toolkits.mplot3d

cell_LRS_mu = 1.34*np.log(10)
cell_LRS_sig = 0.06*np.log(10)  
cell_HRS_mu = 2.62*np.log(10) 
cell_HRS_sig = 0.38*np.log(10)
vol = 0.3 #voltage
RRAM_size = sys.argv[1]


