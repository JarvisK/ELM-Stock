__author__ = 'jarvis'

import numpy as np

class RBFun():

    def __init__(self, P, IW, Bias):
        self.ind = np.ones((len(P), 1))
        '''
        for i in range(1, len(IW)):
            Weight = IW[i]
            WeightMatrix =
        '''