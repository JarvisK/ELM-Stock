__author__ = 'jarvis'

from elm import SimpleELMRegressor
from elm import ELMRegressor
import numpy as np
import pandas as pd

#Read the training csv file
trainData = pd.read_csv('training.csv')

#Make features from training data
#Here I use (close price - open price) and (highest price - lowest price)
X = np.ndarray(shape = (len(trainData) - 3, 6))
X.fill(0)
for i in range(len(trainData) - 3):
    ftr1 = float(trainData['close'][i]) - float(trainData['open'][i])
    ftr2 = float(trainData['close'][i+1]) - float(trainData['open'][i+1])
    ftr3 = float(trainData['close'][i+2]) - float(trainData['open'][i+2])
    ftr4 = float(trainData['highest'][i] - trainData['lowest'][i])
    ftr5 = float(trainData['highest'][i+1] - trainData['lowest'][i+1])
    ftr6 = float(trainData['highest'][i+2] - trainData['lowest'][i+2])
    tmpX = np.array([ftr1, ftr2, ftr3, ftr4, ftr5, ftr6])
    X[i] = tmpX

#Make the output from training data
y = np.zeros(len(trainData) - 3)
for i in range(0, len(trainData) - 3):
    if trainData['changerange'][i+3] >= 0:
        y[i] = 1
    else:
        y[i] = -1

#create ELM instance
reg = ELMRegressor()
#fit the data
reg.fit(X, y)

#Read testing data from testing.csv
testData = pd.read_csv('testing.csv')
pdt = np.zeros(len(testData) - 3)
for i in range(len(testData) - 3):
    ftr1 = float(testData['close'][i]) - float(testData['open'][i])
    ftr2 = float(testData['close'][i+1]) - float(testData['open'][i+1])
    ftr3 = float(testData['close'][i+2]) - float(testData['open'][i+2])
    ftr4 = float(testData['highest'][i] - testData['lowest'][i])
    ftr5 = float(testData['highest'][i+1] - testData['lowest'][i+1])
    ftr6 = float(testData['highest'][i+2] - testData['lowest'][i+2])
    pdt[i] = reg.predict([ftr1, ftr2, ftr3, ftr4, ftr5, ftr6])

#Verify the testing data
#The 0.05 is the threshold to determine rise or full
correct = 0
for i in range(len(testData) - 3):
    if testData['changerange'][i] >=0 and float(pdt[i]) > 0.05:
        correct += 1

#calculate the result
print 'The accuracy is', correct / float(len(testData)) * 100, 'percent'