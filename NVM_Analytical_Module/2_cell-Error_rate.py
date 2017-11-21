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



def print_cur(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/10000)
	plt.plot(x, [func(each) for each in x])

#def print_cur_comb(a, b, func):
def print_cur_comb(a, b, ind, func):
  x = np.arange(a,2*b,(b-a)/10000)
  #print(x[5000])
  #print(func(a, b, x[5000]))
  plt.plot(x, [func(a, b, ind, each) for each in x])

#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(log(x)-m)^2/2sigma^2)
def pdf_log(x, mu, sig): ##resistance
	return 1/x/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sig**2) )

#def pdf_current(x, mu, sig): ##current
#	return (1/x/(sig * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - mu)**2 / (2 * sig**2) )

def pdf_LRS_current(x): ## LRS current
  pdf_L= (1/(x * float(cell_LRS_sig) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - float(cell_LRS_mu) )**2 / (2 * float(cell_LRS_sig) **2) )
  return pdf_L

def pdf_HRS_current(x): ## HRS current
  pdf_H= (1/(x * float(cell_HRS_sig)  * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - float(cell_HRS_mu) )**2 / (2 * float(cell_HRS_sig) **2) ) 
  return pdf_H

def pdf_current_comb(i_L, i_H):
  pdf_L= (1/(i_L * float(cell_LRS_sig) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/i_L) - float(cell_LRS_mu) )**2 / (2 * float(cell_LRS_sig) **2) )
  pdf_H= (1/(i_H * float(cell_HRS_sig)  * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/i_H) - float(cell_HRS_mu) )**2 / (2 * float(cell_HRS_sig) **2) )
  return pdf_L * pdf_H

def combination_compare(AH, AL, BH, BL):
  x = np.arange(0.00001, 0.1, 0.0001)
  y = np.arange(0.00001, 0.1, 0.0001)
    
def integral(a, b, func, mu, sig):
	h = (b-a)/float(1000000)
	xk = [a + i*h for i in range (1,1000000)] 
	xk = np.array(xk)
	pdf = func(xk, mu, sig)
	return integrate.simps(pdf, xk)

def integral_comb_area(a, b, func):
  #x = np.arange(0.00001, 0.1, 0.0001)
  #y = np.arange(0.00001, 0.1, 0.0001)
  #ans = 0
  #for i in x:
  #  for j in y:
  #    ans += 0.0001 * 0.0001* pdf_current_comb(i,j)
  #print(ans)
  return integrate.nquad(func, [[0.000001, 0.1],[0.000001, 0.1]])[0]

'''
#3D curve
x = np.arange(0.00001, 0.1, 0.0001)
y = np.arange(0.00001, 0.1, 0.0001)
x, y = np.meshgrid(x, y)
z = pdf_current_comb(x, y)
ax = plt.subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)  
'''
# a = numb of  LRS cell 
# b = numb of  HRS cell 
#for a in range(int(RRAM_size)+1):
#  b = int(RRAM_size) - a
#  x = np.arange(0.00001, 0.1, 0.0001)
#  y = np.arange(0.00001, 0.1, 0.0001)
#  x, y = np.meshgrid(x, y )
#  z = pdf_current_comb(x, y, a, b)
#  ax = plt.subplot(111,projection='3d')
#  ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)  

## Compare two distribution
for HRS_cell_num_A in range(int(RRAM_size)): ##A
  LRS_cell_num_A = int(RRAM_size) - HRS_cell_num_A
  #print("HA:",HRS_cell_num_A)
  #print("LA:",LRS_cell_num_A)
  for HRS_cell_num_B in range(HRS_cell_num_A+1, int(RRAM_size)+1):
    LRS_cell_num_B = int(RRAM_size) - HRS_cell_num_B
  #  print("HB:",HRS_cell_num_B)
  #  print("LB:",LRS_cell_num_B)
    combination_compare(HRS_cell_num_A, LRS_cell_num_A, HRS_cell_num_B, LRS_cell_num_B )  

##2-cell  
x = np.arange(0.00001, 0.1, 0.0001)
y = np.arange(0.00001, 0.1, 0.0001)
ans =0
ans_1 = 1
ans_2 = 1
for i in x:
  for j in y:
    ans_1 = 0.0001* 0.0001*pdf_LRS_current(i) * pdf_LRS_current(j)
    ans_2 = 0.0001* 0.0001*pdf_HRS_current(i) * pdf_LRS_current(j)
    if ans_1 < ans_2:
      ans += ans_1
print("00->10",ans)
ans = 0
ans_1 = 1
ans_2 = 1
for i in x:
  for j in y:
    ans_1 = 0.0001* 0.0001*pdf_LRS_current(i) * pdf_LRS_current(j)
    ans_2 = 0.0001* 0.0001*pdf_HRS_current(i) * pdf_HRS_current(j)
    if ans_1 < ans_2:
      ans += ans_1
print("00->11",ans)
ans =0
ans_1 = 1
ans_2 = 1
for i in x:
  for j in y:
    ans_1 = 0.0001* 0.0001*pdf_HRS_current(i) * pdf_LRS_current(j)
    ans_2 = 0.0001* 0.0001*pdf_HRS_current(i) * pdf_HRS_current(j)
    if ans_1 < ans_2:
      ans += ans_1
print("10->11",ans)

#print_cur(0.000001,0.1, pdf_LRS_current, float(cell_LRS_mu), float(cell_LRS_sig))
#print_cur(0.000001,0.1, pdf_HRS_current, float(cell_HRS_mu), float(cell_HRS_sig))
#print(integral(0.00001,0.1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig)))
#print(integral(0.00001,0.1, pdf_current, float(cell_HRS_mu), float(cell_HRS_sig)))
#print(integral_comb_area(0.00001,0.1, pdf_current_comb))

#plt.xscale('log')
#plt.axis([0.01,100, 0, 0.2])
plt.show()	



