import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math


def Normal_distribution(mu, variance):
	sigma = math.sqrt(variance)
	x = np.linspace(mu - 5 * sigma, mu + 5* sigma)
	plt.plot(x, mlab.normpdf(x, mu, sigma))
	plt.show()

def pdf(x):
	return math.exp(-(x) ** 2 / (2)) / (math.sqrt(2*math.pi))

def sum_func_xk(xk, func):
	return sum([func(each) for each in xk])

def integral(a, b, n, func):
	h = (b-a)/float(n) #把lowerbnd 和 upperbnd 之間切成n等分
	xk = [a + i*h for i in range (1,n)] 
	return h/2 * (func(a) + 2 * sum_func_xk(xk, func) + func(b))

#http://www.cnblogs.com/zhangte/p/6156212.html
def cal_area(lowerbnd, upperbnd, mu, var): 
	return integral((lowerbnd-mu)/var, (upperbnd-mu)/var, 10000, pdf)

	
print(cal_area(2,3,2,1))
