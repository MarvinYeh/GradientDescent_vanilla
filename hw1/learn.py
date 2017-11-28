import numpy as np
import csv

np.set_printoptions(threshold = np.inf)
data = np.delete(np.transpose(np.genfromtxt("train.csv",delimiter = ",")),0,1)
output = []
out_fin = []


for i in range(0,3):
    data = np.delete(data,0,0)   # delete non-related infos


data_size = 18
for i in range(0,24):
    output.append(np.split(data[i],len(data[i]) / data_size))

for i in range(0,len(output[0])) :
    for j in range(0,24) :
        out_fin.append(output[j][i])
out_fin = np.array(out_fin)
out_fin[np.isnan(out_fin)] = 0