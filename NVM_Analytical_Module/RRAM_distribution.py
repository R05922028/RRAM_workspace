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
vol = 0.3 #voltage
x_axis = []
y_axis = []

def print_cur_current(a, b, func, mu, variance):
	sigma = math.sqrt(variance)
	x = np.arange(a,b,(b-a)/10000)
	return plt.plot(0.3/x, [func(each, mu, variance) for each in x])
	
def print_cur(a, b, func, mu, variance):
	sigma = math.sqrt(variance)
	x = np.arange(a,b,(b-a)/10000)
	return plt.plot(x, [func(each, mu, variance) for each in x])

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(x-m)^2/2sigma^2)
#http://www2.kuas.edu.tw/prof/tsungo/www/Publish/Normal%20Distribution.pdf
def pdf(x, mu, var):
	return math.exp(-(np.log10(x)-mu) ** 2 / (2 * math.sqrt(var) ** 2)) / (math.sqrt(2*math.pi) * math.sqrt(var))


def sum_func_xk(xk, func, mu, var):
	return sum([func(each, mu, var) for each in xk])

def integral(a, b, func, mu, var):
	h = (b-a)/float(10000)
	xk = [a + i*h for i in range (1,10000)] 
	return h/2 * (func(a, mu, var) + 2 * sum_func_xk(xk, func, mu, var) + func(b, mu, var))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, var): 
	return integral(lowerbnd, upperbnd, pdf, mu, var)



#print_cur(1, 25000, pdf, float(cell_LRS_mu) , float(cell_LRS_var))  
plt1 = print_cur_current(1, 25000, pdf, float(cell_LRS_mu) , float(cell_LRS_var))  
plt2 = print_cur_current(1, 25000, pdf, float(cell_HRS_mu) , float(cell_HRS_var))  

#plt.xscale('log')
plt.axis([-0.01, 0.15, 0, 2])
plt.show()	



