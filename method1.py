import numpy as np
from random_hidden_layer import RBFRandomHiddenLayer
from elm import ELMRegressor
from sklearn.utils.extmath import safe_sparse_dot
from oselm import OSELMRegressor

X = [[13,53,64],[42,56,45],[34,53,64],[43,23,12],[34,23,75],[56,34,23]]
y = [3,5,6,5,8,4]

a = OSELMRegressor(6, 'rbf', 6, 1)
a = a.fit(X, y)
print a.predict(X)


r = RBFRandomHiddenLayer(n_hidden=6, gamma=0.1, random_state=0)
elm = ELMRegressor(r)
elm = elm.fit(X, y)
print elm.predict(X)
