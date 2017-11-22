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



def pdf_LRS_current(x): ## LRS current
  pdf_L= (1/(x * float(cell_LRS_sig) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - float(cell_LRS_mu) )**2 / (2 * float(cell_LRS_sig) **2) )
  return pdf_L

def pdf_HRS_current(x): ## HRS current
  pdf_H= (1/(x * float(cell_HRS_sig)  * np.sqrt(2 * np.pi))) * np.exp(-(np.log(vol/x) - float(cell_HRS_mu) )**2 / (2 * float(cell_HRS_sig) **2) ) 
  return pdf_H


'''
#3D curve
x = np.arange(0.00001, 0.1, 0.0001)
y = np.arange(0.00001, 0.1, 0.0001)
x, y = np.meshgrid(x, y)
z = pdf_current_comb(x, y)
ax = plt.subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)  
'''

## Compare two distribution
for HRS_cell_num_A in range(int(RRAM_size)): ##A
  LRS_cell_num_A = int(RRAM_size) - HRS_cell_num_A
  #print("HA:",HRS_cell_num_A)
  #print("LA:",LRS_cell_num_A)
  for HRS_cell_num_B in range(HRS_cell_num_A+1, int(RRAM_size)+1):
    LRS_cell_num_B = int(RRAM_size) - HRS_cell_num_B
  #  print("HB:",HRS_cell_num_B)
  #  print("LB:",LRS_cell_num_B)

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




