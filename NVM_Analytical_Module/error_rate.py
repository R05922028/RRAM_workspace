import pickle as pk
import sys
filename = sys.argv[1]
unit = int(sys.argv[2])
with open(filename,'rb') as f:
    err_file = pk.load(f)
for i in range(unit+1):
    print(i,':',err_file[unit-1][i].count(i))
