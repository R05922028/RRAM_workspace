import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import simps
import numpy as np
from math import *
import matplotlib.mlab as mlab
import math
import sys
from bisect import bisect

cell_LRS_mu = 1.34*np.log(10)
cell_LRS_sig = 0.06*np.log(10) 
cell_HRS_mu = 2.62*np.log(10) 
cell_HRS_sig = 0.38*np.log(10)
vol = 0.3 #voltage
RRAM_size = sys.argv[1]


#f(x) = (1/sigma*math.sqrt(2pi))* exp(-(log(x)-m)^2/2sigma^2)

##------resistance--------##

def pdf_log(x, mu, sig): 
	return 1/x/(sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sig**2) )

##-------current----------##

def pdf_current(x, mu, sig): ##current
	return (1/x/(sig * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - mu)**2 / (2 * sig**2) )

##-------show curve-------##

def print_cur(a, b, func, mu, sigma):
	x = np.arange(a,b,(b-a)/100)
	plt.plot(x, [func(each, mu, sigma) for each in x])

##-------integral---------##

def integral(a, b, func, mu, sig):
  h = (b-a)/float(5000)
  xk = [a + i*h for i in range (1,5000)] 
  xk = np.array(xk)
  pdf = func(xk, mu, sig)
  return integrate.simps(pdf, xk)

#print_cur(0.00001,0.1,pdf_current,cell_LRS_mu, cell_LRS_sig)
#print_cur(0.00001,0.1,pdf_current,cell_HRS_mu, cell_HRS_sig)
print(integral(0.000001,0.1, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig)))


##-----calculate cdf-----##

cdf_LRS_x = []
cdf_LRS_y = []
cdf_HRS_x = []
cdf_HRS_y = []
def cdf_current(a, b, ind):
  h = (b-a)/float(5000)
  xk = [a + i*h for i in range (1,5000)] 
  xk = np.array(xk)
  for i in range(len(xk)):
    if ind == 0:
      prob = integral(a,xk[i], pdf_current, float(cell_LRS_mu), float(cell_LRS_sig))
      cdf_LRS_x.append(xk[i])
      cdf_LRS_y.append(prob)
    else:
      prob = integral(a,xk[i], pdf_current, float(cell_HRS_mu), float(cell_HRS_sig))
      cdf_HRS_x.append(xk[i])
      cdf_HRS_y.append(prob)

cdf_current(0.00001, 0.03, 0) ## LRS
cdf_current(0.00001, 0.03, 1) ## HRS

#------calculate end-----##


#------monte-carlo-------##

N = 160000
#X_total = []
#Y_total = []
Data = []
Data_cnt = 0
#print(current_L)
#print(cdf_HRS_x[0])

def I_total(num_L, num_H):
  total = 0
  for cnt_L in range(num_L):
    ind_L = bisect(cdf_LRS_y, np.random.rand(1))
    total = total + cdf_LRS_x[ind_L-1]
  for cnt_H in range(num_H):
    ind_H = bisect(cdf_HRS_y, np.random.rand(1))
    total = total + cdf_HRS_x[ind_H-1]
  return total 
  
fout = open("distribution_data.csv", 'w')

for num in range(int(RRAM_size)+1):
  a = num
  b = int(RRAM_size) - num
  sample_current = []
  print("Now: a->",a," b->",b)
  for i in range(N):
    cur_total = I_total(a,b)
    sample_current.append(cur_total)

  print("sample-end...")

  x = []
  y = []
  sample_set = set(sample_current)
  #print('sample_set:\n')
  sample_list = [i for i in sample_set]
  for i in range(len(sample_list)):
    x.append(float(sample_list[i]))
    y.append(float(sample_current.count(float(sample_list[i]))/N))
  '''
  #--------output csv-----------##
  fout.write('LRS:'+str(a)+', HRS:'+str(b)+'\n')
  for out_x in range(len(x)):
      fout.write(str(x[out_x])+','+str(y[out_x])+'\n') 
  #-------output csv end--------##
  ''' 

  Data.append([])
  for i in range(len(x)):
    Data[Data_cnt].append((x[i], y[i]))
  Data_cnt += 1
  if num%3 == 0:
    plt.plot(x, y, 'ro', alpha=0.3)
  elif num%3 == 1:
    plt.plot(x, y, 'bo', alpha=0.3)
  elif num%3 == 2:
    plt.plot(x, y, 'go', alpha=0.3)    
  
  print("Plot.end")
Data_sorted = Data
#------monte-carlo-------##
for i in range(len(Data_sorted)):
  Data_sorted[i] = sorted(Data_sorted[i])

#--------Error part--------##
for idx in range(int(RRAM_size)):
  err_rate_left = 0
  err_rate_right = 0
  err_rate = 0
  min_err = 100
  min_ref_cur = 0
  step = np.arange(float(Data_sorted[idx+1][0][0]), float(Data_sorted[idx][len(Data_sorted[0])-1][0]), 0.001)
  for i in range(len(step)):
    err_cnt_left = 0
    err_cnt_right = 0
    for j in range(len(Data[idx])): #left
      if Data[idx][j][0] > step[i]:
        err_cnt_left += 1
    for k in range(len(Data[idx+1])): #right
      if Data[idx+1][k][0] < step[i]:
        err_cnt_right += 1
    err_rate_left = err_cnt_left / len(Data[idx])
    err_rate_right = err_cnt_right / len(Data[idx+1])
    err_rate = err_rate_left + err_rate_right
    if err_rate < min_err:
      min_err_l = err_rate_left
      min_err_r = err_rate_right
      min_err = err_rate
      min_ref_cur = step[i]
  
  print("err_rate :",min_err)
  print(idx,"-->", idx+1, ":", min_err_l)
  print(idx+1,"-->", idx, ":", min_err_r)
  print("ref_cur: ", min_ref_cur)
#--------Error part--------##

plt.show()
