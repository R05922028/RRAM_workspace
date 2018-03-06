from scipy import integrate
from scipy.integrate import simps
import numpy as np
from math import *
import matplotlib.mlab as mlab
import math
import sys
import pickle
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
  
#fout = open("distribution_data.csv", 'w')
Err_list = np.zeros((int(RRAM_size), int(RRAM_size)+1, int(RRAM_size)+1))
RRAM_cnt = int(RRAM_size)

for RRAM_size in range(1, RRAM_cnt+1):
  print("RRAM_size:", RRAM_size)
  Data = []
  Data_cnt = 0
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
  '''
    if num%3 == 0:
      plt.plot(x, y, 'ro', alpha=0.3)
    elif num%3 == 1:
      plt.plot(x, y, 'bo', alpha=0.3)
    elif num%3 == 2:
      plt.plot(x, y, 'go', alpha=0.3)    
    print("Plot.end")
  '''
  Data_sorted = Data
  #------monte-carlo-------##
  for i in range(len(Data_sorted)):
    Data_sorted[i] = sorted(Data_sorted[i])
  
  #--------Error part--------##
  '''
  print(len(Data_sorted[0]))
  if len(Data_sorted[0]) % 2 ==0:
    print("mod=0,",len(Data_sorted[0])/2)
    print(Data_sorted[0][int(len(Data_sorted[0])/2)])
  else:
    print(Data_sorted[0][int((len(Data_sorted[0])-1)/2)])
  '''
  fout = open("error_rate_2_ref.csv", "a+")
  fout.write(str(RRAM_size)+'\n')
  left_ref = 0
  right_ref = 0
  level_SA = 8
   
  if int(RRAM_size) < level_SA+1:
    ref_cnt = int(RRAM_size)
  else:
    ref_cnt = level_SA
  for idx in range(ref_cnt):
    if len(Data_sorted[idx]) % 2 ==0:
      left_ref = float(Data_sorted[idx][int(len(Data_sorted[idx])/2)][0])
    else:
      left_ref = float(Data_sorted[idx][int((len(Data_sorted[idx])-1)/2)][0])
    for idx_2 in range(idx+1,ref_cnt+1):
      if len(Data_sorted[idx_2]) % 2 ==0:
        right_ref = float(Data_sorted[idx_2][int(len(Data_sorted[idx_2])/2)][0])
      else:
        right_ref = float(Data_sorted[idx_2][int((len(Data_sorted[idx_2])-1)/2)][0])
      margin_ref=float((left_ref + right_ref)/2)
    
      cnt_left = 0
      for i in range(len(Data[idx])):
        if Data[idx][i][0] > margin_ref:
          cnt_left += 1
      err_left = cnt_left / len(Data[idx])
      cnt_right = 0
      for j in range(len(Data[idx_2])):
        if Data[idx_2][j][0] < margin_ref:
          cnt_right += 1
      err_right = cnt_right / len(Data[idx_2])
      fout.write(str(idx)+','+str(idx_2)+','+str(err_left)+'\n')
      fout.write(str(idx_2)+','+str(idx)+','+str(err_right)+'\n')
      print(idx,"-->",idx_2,":",err_left)
      print(idx_2,"-->",idx,":",err_right)
      Err_list[RRAM_size-1][idx][idx_2] = float(err_left)
      Err_list[RRAM_size-1][idx_2][idx] = float(err_right)

  if int(RRAM_size)>level_SA:
    for idx_2 in range(ref_cnt+1, int(RRAM_size)+1):
      cnt_right = 0
      cnt_left = 0
      for j in range(len(Data[idx_2])):
        if Data[idx_2][j][0] < margin_ref:
          cnt_right += 1
      cnt_left = len(Data[idx_2]) - cnt_right
      err_left = cnt_left / len(Data[idx_2])
      err_right = cnt_right / len(Data[idx_2])
      fout.write(str(idx_2)+','+str(level_SA-1)+','+str(err_right)+'\n')
      print(idx_2,"-->",str(level_SA-1),":",err_right)
      fout.write(str(idx_2)+','+str(level_SA)+','+str(err_left)+'\n')
      print(idx_2,"-->",str(level_SA),":",err_left)
      Err_list[RRAM_size-1][idx_2][int(level_SA)-1] = float(err_right)
      Err_list[RRAM_size-1][idx_2][int(level_SA)] = float(err_left)
print(Err_list)    
with open('Err_file.pkl', 'wb') as f:
  pickle.dump(Err_list, f) 
#with open('Err_file.pkl', 'rb') as f:
#  mynewlist = pickle.load(f)
#print(mynewlist)

#-------Error part--------##

#plt.savefig('monte-2-ref')