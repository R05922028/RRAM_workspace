import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import sys

def print_cur(a, b, n, func, mu, variance):
	sigma = math.sqrt(variance)
	x = np.arange(a,b,(b-a)/n)
	plt.plot(x, [func(each, mu, variance) for each in x])
	plt.plot(x, mlab.normpdf(x, mu, sigma))
	plt.show()	

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(x-m)^2/2sigma^2)
def pdf(x, mu, var):
	return math.exp(-(x-mu) ** 2 / (2 * math.sqrt(var) ** 2)) / (math.sqrt(2*math.pi) * math.sqrt(var))

def sum_func_xk(xk, func, mu, var):
	return sum([func(each, mu, var) for each in xk])

def integral(a, b, n, func, mu, var):
	h = (b-a)/float(n) #把lowerbnd 和 upperbnd 之間切成n等分
	xk = [a + i*h for i in range (1,n)] 
	return h/2 * (func(a, mu, var) + 2 * sum_func_xk(xk, func, mu, var) + func(b, mu, var))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, var): 
	return integral(lowerbnd, upperbnd, 10000, pdf, mu, var)


cell_LRS_mu = sys.argv[1]
cell_LRS_var = sys.argv[2] 
cell_HRS_mu = sys.argv[3]
cell_HRS_var = sys.argv[4] 
RRAM_size = sys.argv[5]


print_cur(-10, 10, 10000, pdf, 2, 10)  



print(cal_area(-6,10,2,1))

