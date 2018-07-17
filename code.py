import numpy as np
from sys import exit
from random import shuffle
import csv
import discriminantAnal as da

csvFile = open('clean1.tsv', 'rb')
next(csvFile, None)
dataFile = csv.reader(csvFile, delimiter='\t') 
dataDim = 166
sampleSize = 80
totalGroup = 2
poolSampleSize = totalGroup * sampleSize
groupNum = [] 
data = []

for row in dataFile:
    data.append(map(float, row[2:-1]))
    groupNum.append(int(row[-1]))     
secondGroupStart = groupNum.index(0)
secondGroupEnd = secondGroupStart + sampleSize 
data = np.concatenate((data[:][0:sampleSize], data[:][secondGroupStart:secondGroupEnd]), axis=0)
groupNum = sampleSize*[1] + sampleSize*[0]

fold = 5
testSize = int(poolSampleSize/fold)
trainSize = poolSampleSize - testSize 
reducedDimStart = trainSize - 2 

aggErrorRateHaff = []
aggErrorRatePaper = []
dimension = []
rep = 500 
for dim in range(2, reducedDimStart, 5):
    dimension.append(dim)
    for repetition in range(rep):
        index = list(range(poolSampleSize)) 
        shuffle(index)
        misclassifyHaff = 0
        misclassifyPaper = 0
        for i in range(fold):
            testIndex = index[i*testSize : (i+1)*testSize]
            testGroupNum = [groupNum[i] for i in testIndex]
            trainIndex = list(set(index) - set(testIndex)) 
            trainIndex1, trainIndex2 = da.splitTrainDat(trainIndex, groupNum)
    
            test = data[testIndex]
            group1 = data[trainIndex1]
            group2 = data[trainIndex2]
            group = data[trainIndex]
            
            sampleSize1 = np.size(group1, axis=0)
            sampleSize2 = np.size(group2, axis=0)
            mean1 = np.mean(group1, axis=0).reshape(-1,1)
            mean2 = np.mean(group2, axis=0).reshape(-1,1)
            covar1 = np.cov(group1.T)
            covar2 = np.cov(group2.T)
            covarPooled = np.cov(group.T)
            
            dimRedMatPaper = np.concatenate((mean2-mean1, covarPooled), axis=1)
            dimRedMatPaper,a,b = np.linalg.svd(dimRedMatPaper, full_matrices=False) 
            dimRedMatPaper = dimRedMatPaper[:, 0:dim]
            paperTrain1 = group1.dot(dimRedMatPaper) 
            paperTrain2 = group2.dot(dimRedMatPaper) 
            paperTest = test.dot(dimRedMatPaper) 
    
            meanPaper1 = np.mean(paperTrain1, axis=0).reshape(-1, 1)
            covarPaper1 = np.cov(paperTrain1.T)
            meanPaper2 = np.mean(paperTrain2, axis=0).reshape(-1, 1)
            covarPaper2 = np.cov(paperTrain2.T)
    
    
            
            dimRedMat = covar2 - covar1
            dimRedMat, a, b = np.linalg.svd(dimRedMat, full_matrices=False)
            dimRedMat = dimRedMat[:, 0:min(sampleSize1-2, sampleSize2-2)]
            
            newGroup1 = group1.dot(dimRedMat) 
            newGroup2 = group2.dot(dimRedMat)
            newTest = test.dot(dimRedMat)
    
            sampleSize1 = np.size(newGroup1, axis=0)
            dataDim1 = np.size(newGroup1, axis=1) 
            sampleSize2 = np.size(newGroup2, axis=0)
            dataDim2 = np.size(newGroup2, axis=1) 
            
            newMean1 = np.mean(newGroup1, axis=0).reshape(-1, 1)
            newCovar1 = np.cov(newGroup1.T)
            newMean2 = np.mean(newGroup2, axis=0).reshape(-1, 1)
            newCovar2 = np.cov(newGroup2.T)
            
            sTildeInv1 = da.sTildeInv(newCovar1, dataDim1, sampleSize) 
            sTildeInv2 = da.sTildeInv(newCovar2, dataDim2, sampleSize) 
            dimRedMat = np.concatenate((sTildeInv1.dot(newMean1)-sTildeInv2.dot(newMean2),
                        newCovar2-newCovar1), axis=1)
            dimRedMat,a,b = np.linalg.svd(dimRedMat)
            dimRedMat = dimRedMat[:, 0:dim]
    
            trainHaff1 = newGroup1.dot(dimRedMat)
            trainHaff2 = newGroup2.dot(dimRedMat)
            testHaff = newTest.dot(dimRedMat)
            
            meanHaff1 = np.mean(trainHaff1, axis=0).reshape(-1, 1)
            covarHaff1 = np.cov(trainHaff1.T)
            meanHaff2 = np.mean(trainHaff2, axis=0).reshape(-1, 1)
            covarHaff2 = np.cov(trainHaff2.T)
            for j in range(testSize):
                distHaff1 = da.qdf(testHaff[j, : ].reshape(-1, 1), covarHaff1, meanHaff1) 
                distHaff2 = da.qdf(testHaff[j, : ].reshape(-1, 1), covarHaff2, meanHaff2) 
                if da.classify(distHaff1, distHaff2) != testGroupNum[j]:
                    misclassifyHaff += 1
                distPaper1 = da.qdf(paperTest[j, : ].reshape(-1, 1), covarPaper1, meanPaper1)
                distPaper2 = da.qdf(paperTest[j, : ].reshape(-1, 1), covarPaper2, meanPaper2)
                if da.classify(distPaper1, distPaper2) != testGroupNum[j]:
                    misclassifyPaper += 1
    
        aggErrorRateHaff.append(misclassifyHaff/float(fold*testSize))         
        aggErrorRatePaper.append(misclassifyPaper/float(fold*testSize))         

aggErrorRateHaff = np.array(aggErrorRateHaff).reshape((rep, -1), order='F')
aggErrorRatePaper = np.array(aggErrorRatePaper).reshape((rep, -1), order='F')


np.save('dataFile', aggErrorRateHaff)
np.save('dataFile2', aggErrorRatePaper)
np.save('dimensions', dimension)
