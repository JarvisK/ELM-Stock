__author__ = 'jarvis'

from random_hidden_layer import RBFRandomHiddenLayer
from sklearn.utils.extmath import safe_sparse_dot
from scipy.linalg import pinv2
import numpy as np

class OSELMRegressor():

    def __init__(self, nHiddenNeurons, ActivationFunction, N0, Block):
        """

        :param nHiddenNeurons:?
        :param ActivationFunction:?
        :param N0: Number of initial training data used in the initial phase of OSELM, which is not less than the number of hidden neurons
        :param Block: Size of block of data learned by OSELM in each step.
        :return:?
        """
        self.nHiddenNeurons = nHiddenNeurons
        self.ActivationFunction = ActivationFunction
        self.N0 = N0
        self.Block = Block

        self.Bias = 0

    def fit(self, X, y):
        """

        :param X: feature
        :param y: target
        :return:
        """
        X = np.array(X)
        y = np.array(y)
        nInputNeurons = len(X[0]) #how many element in one feature
        nTrainingData = len(X)
        P0 = X[0:self.N0]
        T0 = y[0:self.N0]
        H0 = None

        if self.ActivationFunction == 'rbf':
            H0 = RBFRandomHiddenLayer(n_hidden=len(P0[0]), gamma=0.1, random_state=0).fit_transform(P0)
            print H0

        M = np.linalg.pinv(safe_sparse_dot(H0.transpose(), H0))
        self.beta = safe_sparse_dot(np.linalg.pinv(H0), T0)

        for i in range(self.N0, nTrainingData, self.Block):
            if(i + self.Block) > nTrainingData:
                Pn = X[i:nTrainingData]
                Tn = y[i:nTrainingData]
                self.Block = len(Pn)
            else:
                Pn = X[i:(i+self.Block)]
                Tn = y[i:(i+self.Block)]

            if self.ActivationFunction == 'rbf':
                H = RBFRandomHiddenLayer(n_hidden=len(Pn[0]), gamma=0.2, random_state=0).fit_transform(Pn)

            tempM = np.linalg.inv(np.eye(self.Block) + (H.dot(M).dot(H.transpose())))
            tempM = M.dot(H.transpose()).dot(tempM).dot(H).dot(M)
            M = M - tempM
            self.beta = self.beta + (M.dot(H.transpose()).dot(Tn - (H.dot(self.beta))))

            #M = M - M * H.transpose() * pinv2((np.eye(self.Block) + H * M * H.transpose())) * H * M
            #self.beta = self.beta + M * H.transpose() * (Tn - H * self.beta)

        return self

    def predict(self, X):
        X = np.array(X)

        if self.ActivationFunction == 'rbf':
            HResult = RBFRandomHiddenLayer(n_hidden=len(X[0]), gamma=0.2, random_state=0).fit_transform(X)

        #return HResult * self.beta
        return safe_sparse_dot(HResult, self.beta)