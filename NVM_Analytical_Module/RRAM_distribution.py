import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import simps
import numpy as np
import matplotlib.mlab as mlab
import math
import sys

cell_LRS_mu = 1.34*np.log(10)
cell_LRS_sig = 0.06*np.log(10)  
cell_HRS_mu = 2.62*np.log(10) 
cell_HRS_sig = 0.38*np.log(10)
vol = 0.3 #voltage
RRAM_size = sys.argv[1]
max_prob = 0
new_mu = 0


def print_cur(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(x, [func(each, mu, sigma) for each in x])
def cal_new_mu(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/10000)
	[func(each, mu, sigma) for each in x]
	global max_prob
	max_prob = 0
	global new_mu
	return new_mu

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(log(x)-m)^2/2sigma^2)
def pdf_log(x, mu, sig):
	return 1/x/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sig**2) )

def pdf_current(x, mu, sig):
	return (1/(x)/(sig * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - mu)**2 / (2 * sig**2) )


def integral(a, b, func, mu, sig):
	h = (b-a)/float(1000000)
	xk = [a + i*h for i in range (1,1000000)] 
	xk = np.array(xk)
	pdf = func(xk, mu, sig)
	return integrate.simps(pdf, xk)



#print_cur(1,50, pdf_log, float(cell_LRS_mu), float(cell_LRS_sig))
print_cur(0.000001,0.1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig))
print_cur(0.000001,0.1, pdf_current, float(cell_HRS_mu), float(cell_HRS_sig))
#print_cur(0.001,1, pdf_log, float(cell_LRS_current_mu), float(cell_LRS_sig))

#print_cur(1,1000, pdf_log, float(cell_LRS_mu), float(cell_LRS_sig))
#print_cur(0.00001,0.1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig))
#print_cur(0.00001,0.1, pdf_current, float(cell_HRS_mu), float(cell_HRS_sig))
print(cell_LRS_mu)
print(integral(0.00001,1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig)))

plt.xscale('log')
#plt.axis([0.01,100, 0, 0.2])
plt.show()	



