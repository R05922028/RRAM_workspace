import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import sys

cell_LRS_mu = sys.argv[1]
cell_LRS_var = sys.argv[2] 
cell_HRS_mu = sys.argv[3]
cell_HRS_var = sys.argv[4] 
RRAM_size = sys.argv[5]

def print_cur(a, b, func, mu, variance):
	sigma = math.sqrt(variance)
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(x, [func(each, mu, variance) for each in x])
	plt.plot(x, mlab.normpdf(x, mu, sigma))

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(x-m)^2/2sigma^2)
#http://www2.kuas.edu.tw/prof/tsungo/www/Publish/Normal%20Distribution.pdf
def pdf(x, mu, var):
	return math.exp(-(x-mu) ** 2 / (2 * math.sqrt(var) ** 2)) / (math.sqrt(2*math.pi) * math.sqrt(var))

def sum_func_xk(xk, func, mu, var):
	return sum([func(each, mu, var) for each in xk])

def integral(a, b, func, mu, var):
	h = (b-a)/float(10000)
	xk = [a + i*h for i in range (1,10000)] 
	return h/2 * (func(a, mu, var) + 2 * sum_func_xk(xk, func, mu, var) + func(b, mu, var))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, var): 
	return integral(lowerbnd, upperbnd, pdf, mu, var)




print_cur(0, 5, pdf, float(cell_LRS_mu) , float(cell_LRS_var))  
print_cur(0, 5, pdf, float(cell_HRS_mu) , float(cell_HRS_var))  
plt.show()	



print(cal_area(-6,10,2,1))

