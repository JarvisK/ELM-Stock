import numpy as np
from sklearn.kernel_approximation import RBFSampler


a = [[1,2,3],[2,3,4],[3,4,5]]
b = [5,8]
#a = np.c_[a, np.ones(2)]

bb = RBFSampler(n_components=3)

c = np.random.rand(3, 2) * 2 - 1

x = bb.fit_transform(a, b)
#b = np.ones((len(a), 1))

print a[0:2]