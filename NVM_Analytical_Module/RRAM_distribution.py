import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import sys

cell_LRS_mu = 1.34
cell_LRS_sig = 0.06 
cell_HRS_mu = 2.62
cell_HRS_sig = 0.38 
vol = 0.3 #voltage
RRAM_size = sys.argv[1]


def print_cur_current(a, b, func, mu, sig):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(0.3/x, [func(each, mu, sig) for each in x])


def print_cur(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(x, [func(each, mu, sigma) for each in x])

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(log(x)-m)^2/2sigma^2)
#def pdf_resistance(x, mu, sig):
#	return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log10(x) - mu)**2 / (2 * sig**2) )

def pdf_current(x, mu, sig):
	return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log10(vol/x) - mu)**2 / (2 * sig**2) )

def sum_func_xk(xk, func, mu, sig):
	return sum([func(each, mu, sig) for each in xk])

def integral(a, b, func, mu, sig):
	h = (b-a)/float(10000)
	xk = [a + i*h for i in range (1,10000)] 
	return h/2 * (func(a, mu, sig) + 2 * sum_func_xk(xk, func, mu, sig) + func(b, mu, sig))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, sig): 
	return integral(lowerbnd, upperbnd, pdf_current, mu, sig)


for i in range(int(RRAM_size)+1):
  alpha = i
  beta = int(RRAM_size) - alpha
  alpha /= int(RRAM_size)
  beta /= int(RRAM_size)
  mu = alpha * float(cell_LRS_mu) + beta * float(cell_HRS_mu)
  sigma = ((alpha ** 2) * (float(cell_LRS_sig) ** 2) + (beta ** 2) * (float(cell_HRS_sig) ** 2)) ** 0.5
  print_cur(0.00001, 0.1, pdf_current, mu, sigma) 
  

#print_cur(0.00001, 0.1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig))  
#print_cur(0.00001, 0.1, pdf_current, float(cell_HRS_mu), float(cell_HRS_sig))  
#print(cal_area(0.007,0.008,1.34,0.06))

#plt.xscale('log')
plt.axis([0.00001,0.02, 0, 8])
plt.show()	



