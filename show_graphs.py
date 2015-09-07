import matplotlib.pyplot as plt
import numpy as np
from LinearLeastSquares import LinearLeastSquares
import shape_data
from time import time

t0 = time()
protocol_result_file = "E:/Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
              "Project/protocol_result_data.dat"
total_result_file = "E:/Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
              "Project/total_result_data.dat"
raw_data_file = "E:\Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
                "Project/PAMAP2_Dataset/Protocol/subject101.dat"
raw_data_file2 = "E:\Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
                "Project/PAMAP2_Dataset/Protocol/subject102.dat"
raw_data_file3 = "E:\Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
                "Project/PAMAP2_Dataset/Protocol/subject103.dat"
raw_data_file4 = "E:\Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
                "Project/PAMAP2_Dataset/Protocol/subject104.dat"
raw_data_file5 = "E:\Documents/Passau - 2014 - 2015/Machine learnig and context recognition/" \
                "Project/PAMAP2_Dataset/Protocol/subject105.dat"
data_set = np.loadtxt(protocol_result_file, delimiter=" ")
raw_data_set = np.loadtxt(raw_data_file, delimiter=" ")
raw_data_set2 = np.loadtxt(raw_data_file2, delimiter=" ")
raw_data_set3 = np.loadtxt(raw_data_file3, delimiter=" ")
raw_data_set4 = np.loadtxt(raw_data_file4, delimiter=" ")
raw_data_set5 = np.loadtxt(raw_data_file5, delimiter=" ")
t1 = time()-t0
print("t1 = ", t1)
"""

energy_data = data_set[:, 4]
std_data = data_set[:, 16]
std_data2 = data_set[:, 17]
std_data3 = data_set[:, 18]
"""

raw_data = raw_data_set[:, 2]
raw_data = raw_data[~np.isnan(raw_data)]

raw_data2 = raw_data_set2[:, 2]
raw_data2 = raw_data2[~np.isnan(raw_data2)]

raw_data3 = raw_data_set3[:, 2]
raw_data3 = raw_data3[~np.isnan(raw_data3)]

raw_data4 = raw_data_set4[:, 2]
raw_data4 = raw_data4[~np.isnan(raw_data4)]

raw_data5 = raw_data_set5[:, 2]
raw_data5 = raw_data5[~np.isnan(raw_data5)]

raw_data1 = raw_data_set[:, 2]
# raw_data2 = data_set[:, 3]
#raw_data3 = raw_data_set[:, 23]
print(len(raw_data1))
print(len(raw_data2))
plt.figure(facecolor='white')
plt.plot(raw_data, 'b')
plt.plot(raw_data4, 'g')
# plt.plot(raw_data2, 'r')
# plt.plot(raw_data3, 'y')
plt.show()
"""
plt.subplot(211)
plt.xlabel('Chest IMU data points')
plt.ylabel('acceleration')
plt.title('raw data')
plt.plot(raw_data, 'b')
plt.subplot(212)
plt.xlabel('Segmented Chest IMU data points')
plt.ylabel('standard deviation')
plt.title('processed data ')
plt.plot(raw_data2, 'b')
plt.show()

t2 = time()-t1
print("t2 = ", t2)
#plt.subplot(311)
#plt.plot(raw_data1, 'b')
#plt.subplot(312)
#plt.plot(raw_data2 ,  'b')
#plt.subplot(313)
#plt.plot(raw_data1, 'b')
t3 = time()-t2
print("t3 = ", t3)
print("total time = ", time())
#plt.show()

"""
