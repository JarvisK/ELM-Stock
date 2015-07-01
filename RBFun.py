__author__ = 'jarvis'

import numpy as np

class RBFun():

    def __init__(self, P, IW, Bias):
        self.ind = np.zeros(len(P), 1)
        self.ind.fill(1)

        for i in range(1, len(IW)):
            Weight = 