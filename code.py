import numpy as np
import sklearn as sk

sampleSize1 = 70
sampleSize2 = 50
dataDim = 200

mean1 = np.random.rand(dataDim, 1)*200 + 100  
mean2 = np.random.rand(dataDim, 1)*200 + 300
covar1 = sk.datasets.make_spd_matrix(dataDim)
covar2 = covar1 + np.diag(np.random.rand(dataDim, 1))

group1 = np.random.multivariate_normal(mean1, covar1)
group2 = np.random.multivariate_normal(mean2, covar2)

