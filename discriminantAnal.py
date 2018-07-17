import sys
import numpy as np
import scipy as sp
import math 
from random import shuffle

def sTildeInv(covar, dataDim, sampleSize):
    firstPart = (1-tUEye(covar, dataDim, sampleSize)) * (sampleSize-dataDim-2) * np.linalg.inv(covar)
    secondPart = tUEye(covar, dataDim, sampleSize) * (dataDim*sampleSize-dataDim-2) / \
        np.trace(covar) * np.eye(dataDim)
    return firstPart + secondPart
    

def tUEye(covar, dataDim, sampleSize):
    coeff = float(min(1, 4*(dataDim**2-1))/((sampleSize-dataDim-2)*dataDim**2))
    result = coeff * uEye(covar, dataDim)**(1.0/dataDim)
    return result

def uEye(covar, dataDim):
    result = dataDim * np.linalg.det(covar)**(1.0/dataDim)
    result = result / np.trace(covar)
    return result

def splitTrainDat(train, groupNum):
    group1 = [] 
    group2 = [] 
    for i in map(int, train):
        if groupNum[i] == 1:
            group1.append(i)
        else: 
            group2.append(i)
    return (group1, group2)
    

def classify(dist1, dist2):
    groupNum = 1
    if dist2 <= dist1:
        groupNum = 2
    return groupNum

def qdf(data, sampleCov, sampleMean):
    distance = math.log(np.linalg.det(np.matrix(sampleCov))) + \
        (data-sampleMean).T.dot(np.linalg.inv(np.matrix(sampleCov))).dot(data-sampleMean)
    return distance

