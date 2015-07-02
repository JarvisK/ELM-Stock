__author__ = 'jarvis'

from sklearn.kernel_approximation import RBFSampler
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
        self.ptime = time.clock()
        self.Bias = 0

    def calculate(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        if self.ActivationFunction == "rbf":
            rbf = RBFSampler(gamma=2, n_components=len(X[0]))
            H0 = rbf.fit_transform(X[0:self.N0], y)

        M = np.linalg.pinv(H0.transpose() * H0)
        beta = np.linalg.pinv(H0) * y[0:self.N0]

        for i in range(self.N0, len(X), self.Block):
            if (i + self.Block - 1) > len(y):
                Pn = X[i:len(X)]
                Tn = y[i:len(X)]
                self.Block = len(Pn)
                V = 0;
            else:
                Pn = X[i:(i+self.Block-1)]
                Tn = y[i:(i+self.Block-1)]

            if self.ActivationFunction == "rbf":
                H = RBFSampler(gamma=2, n_components=len(Pn[0])).fit_transform(Pn, Tn)

            M = M - M * H.transpose() * np.linalg.inv((np.eye(self.Block) + H * M * H.transpose())) * H * M
            beta = beta + M * H.transpose() * (Tn - H * beta)

        if self.ActivationFunction == "rbf":
            HTrain = RBFun()

        Y = HTrain * beta

        if self.ActivationFunction == "rbf":
