import numpy as np
import sklearn as sk
import csv

csvFile = open('clean1.tsv', 'rb')
next(csvFile, None)
dataFile = csv.reader(csvFile, delimiter='\t') 
dataDim = 166
sampleSize = 80
group = [] 
data = []

for row in dataFile:
    data.append(map(float, row[2:-1]))
    group.append(int(row[-1]))     
secondGroupStart = group.index(0)
secondGroupEnd = secondGroupStart + sampleSize 
data = np.concatenate((data[0:sampleSize][:], data[secondGroupStart:secondGroupEnd][:]), axis=0)
group = sampleSize*[1] + sampleSize*[0]
