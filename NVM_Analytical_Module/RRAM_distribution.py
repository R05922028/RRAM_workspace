import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import sys

cell_LRS_mu = sys.argv[1]
cell_LRS_sig = sys.argv[2] 
cell_HRS_mu = sys.argv[3]
cell_HRS_sig = sys.argv[4] 
RRAM_size = sys.argv[5]
vol = 0.3 #voltage
#x_axis = []
y_axis = []
LRS = []


def print_cur_current(a, b, func, mu, sig):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(0.3/x, [func(each, mu, sig) for each in x])
	#x_axis = 0.3/x
	y_axis = [func(each, mu, sig) for each in x]
	return y_axis


def print_cur(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(x, [func(each, mu, sigma) for each in x])

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(x-m)^2/2sigma^2)
#http://www2.kuas.edu.tw/prof/tsungo/www/Publish/Normal%20Distribution.pdf
def pdf(x, mu, sig):
	return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sig**2) )
	#return math.exp(-(np.log10(x)-mu) ** 2 / (2 * (sig ** 2))) / (math.sqrt(2*math.pi) * sig)
def pdf_inverse(x, mu, sig):
	return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sig**2) )

def sum_func_xk(xk, func, mu, sig):
	return sum([func(each, mu, sig) for each in xk])

def integral(a, b, func, mu, sig):
	h = (b-a)/float(10000)
	xk = [a + i*h for i in range (1,10000)] 
	return h/2 * (func(a, mu, sig) + 2 * sum_func_xk(xk, func, mu, sig) + func(b, mu, sig))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, sig): 
	return integral(lowerbnd, upperbnd, pdf, mu, sig)

distr = []
#print_cur(1, 10000, pdf, float(cell_LRS_mu)*np.log(10) , float(cell_LRS_sig)*np.log(10))  
#print_cur(1, 10000, pdf, float(cell_HRS_mu)*np.log(10) , float(cell_HRS_sig)*np.log(10))  
print_cur(1, 1000000, pdf, (np.log(0.3)-float(cell_LRS_mu)*np.log(10)) , float(cell_LRS_sig)*np.log(10))  
#print_cur(1, 10000, pdf, float(cell_HRS_mu)*np.log(10) , float(cell_HRS_sig)*np.log(10))  
#distr.append(print_cur_current(1, 10000, pdf, float(cell_LRS_mu) , float(cell_LRS_sig)))
#distr.append(print_cur_current(1, 10000, pdf, float(cell_HRS_mu) , float(cell_HRS_sig)))
#print(len(distr))
#print(len(distr[0]))

plt.xscale('log')
#plt.axis([0,100000, 0, 8])
plt.show()	



