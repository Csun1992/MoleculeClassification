import numpy as np
from matplotlib import pyplot as plt
from sys import exit

dimension = np.load('dimensions.npy')
name = 'Dim '
num = list(map(str, dimension))
label = list(map(lambda y:name+y, num))
data1 = np.load('dataFile.npy')
print data1
print '\n'
data2 = np.load('dataFile2.npy')
print data2

plt.figure(1)
plt.boxplot(data1)

plt.xticks(range(1, np.size(data1, axis=1)+1), label)
plt.xlabel('Reduced Dimension')
plt.ylabel('Error Rate')
plt.title('Error Rate Estimate by Our Methods')

plt.figure(2)
plt.boxplot(data2)

plt.xticks(range(1, np.size(data2, axis=1)+1), label)
plt.xlabel('Reduced Dimension')
plt.ylabel('Error Rate')
plt.title('Error Rate Estimate by Methods from the Paper')

plt.show()
