__author__ = 'jarvis'

from elm import SimpleELMRegressor
from elm import ELMRegressor
from random_hidden_layer import SimpleRandomHiddenLayer
from random_hidden_layer import RBFRandomHiddenLayer
from sklearn.linear_model import LogisticRegression
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
    ftr4 = float(trainData['highest'][i] - float(trainData['lowest'][i]))
    ftr5 = float(trainData['highest'][i+1] - float(trainData['lowest'][i+1]))
    ftr6 = float(trainData['highest'][i+2] - float(trainData['lowest'][i+2]))
    tmpX = np.array([ftr1, ftr2, ftr3, ftr4, ftr5, ftr6])
    X[i] = tmpX

#Make the output from training data
y = np.zeros(len(trainData) - 3)
for i in range(0, len(trainData) - 3):
    if trainData['changerange'][i+3] >= 0:
        y[i] = 1
    else:
        y[i] = -1

srhl_tanh = SimpleRandomHiddenLayer(n_hidden=100, activation_func='tanh', random_state=0)
srhl_rbf = RBFRandomHiddenLayer(n_hidden=200*2, gamma=0.1, random_state=0)
#create ELM instance
reg = ELMRegressor(srhl_tanh)
#fit the data
reg.fit(X, y)

#Read testing data from testing.csv
testData = pd.read_csv('testing.csv')
pdt = np.zeros(len(testData) - 3)
for i in range(len(testData) - 3):
    ftr1 = float(testData['close'][i]) - float(testData['open'][i])
    ftr2 = float(testData['close'][i+1]) - float(testData['open'][i+1])
    ftr3 = float(testData['close'][i+2]) - float(testData['open'][i+2])
    ftr4 = float(testData['highest'][i] - float(testData['lowest'][i]))
    ftr5 = float(testData['highest'][i+1] - float(testData['lowest'][i+1]))
    ftr6 = float(testData['highest'][i+2] - float(testData['lowest'][i+2]))
    pdt[i] = reg.predict([ftr1, ftr2, ftr3, ftr4, ftr5, ftr6])

print pdt
#Verify the testing data
#The 0.01 is the threshold to determine rise or full
correct = 0
for i in range(len(testData) - 3):
    if testData['changerange'][i] >=0 and float(pdt[i]) >= 0.09:
        correct += 1
    elif testData['changerange'][i] < 0 and float(pdt[i]) < 0.09:
        correct += 1

#calculate the result
print 'The accuracy is', float(correct / float(len(testData))) * 100, 'percent'