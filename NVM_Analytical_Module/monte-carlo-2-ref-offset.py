from scipy import integrate
from scipy.integrate import simps
import numpy as np
from math import *
import matplotlib.mlab as mlab
import math
import sys
import pickle
from bisect import bisect
import pickle as pk
import random


LRS_mu = 1.34
LRS_sig = 0.06
HRS_mu = 2.62
HRS_sig = 0.38
cell_LRS_mu = LRS_mu*np.log(10)
cell_LRS_sig = LRS_sig*np.log(10) 
cell_HRS_mu = HRS_mu*np.log(10) 
cell_HRS_sig = HRS_sig*np.log(10)
vol = 0.6 #voltage
RRAM_size = sys.argv[1]
sensing_offset = 0 #v

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
  h = (b-a)/float(10000)
  xk = [a + i*h for i in range (1,10000)] 
  xk = np.array(xk)
  pdf = func(xk, mu, sig)
  return integrate.simps(pdf, xk)

#print_cur(0.00001,0.1,pdf_current,cell_LRS_mu, cell_LRS_sig)
#print_cur(0.00001,0.1,pdf_current,cell_HRS_mu, cell_HRS_sig)
print(integral(0.018,0.0415, pdf_current, float(cell_LRS_mu), float(cell_LRS_sig)))
print(integral(0,0.02, pdf_current, float(cell_HRS_mu), float(cell_HRS_sig)))



##-----calculate cdf-----##

cdf_LRS_x = []
cdf_LRS_y = []
cdf_HRS_x = []
cdf_HRS_y = []
def cdf_current(a, b, ind):
  h = (b-a)/float(10000)
  xk = [a + i*h for i in range (1,10000)] 
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
#LRS current margin
lower_bound_LRS = vol/(10**(LRS_mu+3*LRS_sig))
higher_bound_LRS = vol/(10**(LRS_mu-3*LRS_sig))
#HRS current margin
lower_bound_HRS = vol/(10**(HRS_mu+3*HRS_sig))
higher_bound_HRS = vol/(10**(HRS_mu-3*HRS_sig))

cdf_current(lower_bound_LRS, higher_bound_LRS, 0) ## LRS
cdf_current(lower_bound_HRS, higher_bound_HRS, 1) ## LRS
#------calculate end-----##

#------monte-carlo-------##

N = 32000
#X_total = []
#Y_total = []
#print(current_L)
#print(cdf_HRS_x[0])

def I_total(num_L, num_H):
  total = 0
  for cnt_L in range(num_L):
    seed = np.random.rand(1)
    ind_L = bisect(cdf_LRS_y, seed*0.997)
    total = total + cdf_LRS_x[ind_L-1]
  for cnt_H in range(num_H):
    seed = np.random.rand(1)
    ind_H = bisect(cdf_HRS_y, seed*0.997)
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
  level_SA = (2**4)-1
   
  if int(RRAM_size) < level_SA+1:
    ref_cnt = int(RRAM_size)
  else:
    ref_cnt = level_SA
  for idx in range(ref_cnt):
    idx_2 = idx+1
    if len(Data_sorted[idx]) % 2 ==0:
      left_ref = float(Data_sorted[idx][int(len(Data_sorted[idx])/2)][0])
    else:
      left_ref = float(Data_sorted[idx][int((len(Data_sorted[idx])-1)/2)][0])
    if len(Data_sorted[idx_2]) % 2 ==0:
      right_ref = float(Data_sorted[idx_2][int(len(Data_sorted[idx_2])/2)][0])
    else:
      right_ref = float(Data_sorted[idx_2][int((len(Data_sorted[idx_2])-1)/2)][0])
    if idx != ref_cnt-1:
      idx_3 = idx_2 + 1 
      if len(Data_sorted[idx_3]) % 2 ==0:
        next_ref = float(Data_sorted[idx_3][int(len(Data_sorted[idx_3])/2)][0])
      else:
        next_ref = float(Data_sorted[idx_3][int((len(Data_sorted[idx_3])-1)/2)][0])
    else:
      next_ref = 10000 #dont-care
    if idx != 0:
      idx_0 = idx - 1
      if len(Data_sorted[idx_0]) % 2 ==0:
        front_ref = float(Data_sorted[idx_0][int(len(Data_sorted[idx_0])/2)][0])
      else:
        frontt_ref = float(Data_sorted[idx_0][int((len(Data_sorted[idx_0])-1)/2)][0])
    else:
      front_ref = 0     
    margin_ref=float((left_ref + right_ref)/2)- sensing_offset
    next_margin_ref=float((next_ref + right_ref)/2)- sensing_offset
    front_margin_ref=float((front_ref + left_ref)/2)-sensing_offset
    print(margin_ref)
    
    for current in range(RRAM_size+1):
      cnt_left = 0
      cnt_right = 0
      err_left = 0
      err_right = 0
      if current < idx_2:
        for i in range(len(Data[current])):
          if Data[current][i][0] > margin_ref and Data[current][i][0] < next_margin_ref:
            cnt_left += 1
        err_left = cnt_left / len(Data[current])
        fout.write(str(current)+','+str(idx_2)+','+str(err_left)+'\n')
        print(current,"-->",idx_2,":",err_left)
        Err_list[RRAM_size-1][current][idx_2] = float(err_left)
      else:
        for j in range(len(Data[current])):
          if Data[current][j][0] < margin_ref and Data[current][j][0] > front_margin_ref:
            cnt_right += 1
        err_right = cnt_right / len(Data[current])
        fout.write(str(current)+','+str(idx)+','+str(err_right)+'\n')
        print(current,"-->",idx,":",err_right)
        Err_list[RRAM_size-1][current][idx] = float(err_right)

  if int(RRAM_size)>level_SA:
    for idx_2 in range(ref_cnt+1, int(RRAM_size)+1):
      cnt_right = 0
      cnt_left = 0
      for j in range(len(Data[idx_2])):
        if Data[idx_2][j][0] < margin_ref and Data[idx_2][j][0] > front_margin_ref:
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
  for row in range(RRAM_size+1):
    summation = sum(Err_list[RRAM_size-1][row])
    Err_list[RRAM_size-1][row][row] = 1-summation
print(Err_list)    
for i in range(Err_list.shape[0]):
  for j in range(i+2):
    prob = Err_list[i][j][0]
    Err_list[i][j][0] = 0
    for k in range(i+2):
      prob += Err_list[i][j][k]
      Err_list[i][j][k] = prob
print(Err_list)    

#New format
err = Err_list
m = []

for u in range(int(sys.argv[1])):
    m.append([])
    for i in range(int(sys.argv[1])+1):
        m[u].append([0]*int(err[u][i][0]*100.))
        for j in range(1, int(sys.argv[1])+1):
            m[u][i] += [j]*int((err[u][i][j]-err[u][i][j-1])*100.)
        m[u][i] += [i] * (100-len(m[u][i]))
        random.shuffle(m[u][i])
pk.dump(m, open('Err_file_mean_2.62_var_1_SA_4.p', 'wb'))

with open('Err_file_mean_2.62_var_1_SA_4.pkl', 'wb') as f:
  pickle.dump(Err_list, f) 

#with open('Err_file.pkl', 'rb') as f:
#  mynewlist = pickle.load(f)
#print(mynewlist)

#-------Error part--------##

#plt.savefig('monte-2-ref')
