import numpy as np
import sklearn as sk
import csv

csvFile = open('clean1.tsv', 'rb')
next(csvFile, None)
dataFile = csv.reader(csvFile, delimiter='\t') 
dataDim = 166
sampleSize = 80
totalGroup = 2
groupNum = [] 
data = []

for row in dataFile:
    data.append(map(float, row[2:-1]))
    groupNum.append(int(row[-1]))     
secondGroupStart = group.index(0)
secondGroupEnd = secondGroupStart + sampleSize 
data = np.concatenate((data[0:sampleSize][:], data[secondGroupStart:secondGroupEnd][:]), axis=0)
groupNum = sampleSize*[1] + sampleSize*[0]
group1 = data[0:sampleSize]
group2 = data[sampleSize:0]
index = list(range(totalGroup*sampleSize)) 

