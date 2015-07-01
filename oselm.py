__author__ = 'jarvis'

import random

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

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        nInputNeurons = len(X[0])
        IW =

        if(self.ActivationFunction == "rbf"):
            self.Bias = rand
