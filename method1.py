from elm import ELMRegressor
import numpy as np
import pandas as pd
from pandas import *
from sklearn.datasets import make_circles

#print make_circles(n_samples=10)[1]

trainData = pd.read_csv('training.csv')
DATA_SIZE = len(trainData) / 3

X = np.ndarray(shape = (DATA_SIZE, 3), dtype=np.float64)
X.fill(0)
for i, j in zip(range(0, len(trainData), 3), range(DATA_SIZE)):
    if(i+3 > len(trainData)):
        break
    X[j] = trainData['open'].values[range(i, i+3)]
#print X

y = np.zeros(DATA_SIZE)
for i, j in zip(range(3, len(trainData), 4), range(DATA_SIZE)):
    if(trainData['changerange'][i] > 0):
        y[j] = 1
    else:
        y[j] = -1
#print y
#print X[range(1650, 1714)]
#print y[range(1650, 1714)]

'''
reg = ELMRegressor()
reg.fit(X, y)
print reg.predict([140.5, 137.5, 133.5])
'''
'''
testData = pd.read_csv('testing.csv')
pdt = np.zeros(len(testData))
for i in range(len(testData) - 3):
    pdt[i] = reg.predict([testData['open'][i], testData['open'][i+1], testData['open'][i+2]])

print pdt
'''