__author__ = 'jarvis'
#SimpleRandomHiddenLayer(node number, function, random_state=0)
#RBFRandomHiddenLayer(node number, gamma, random_state = 0)
#use ELMClassifier(use above function)

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from random_hidden_layer import SimpleRandomHiddenLayer
from elm import ELMClassifier
import numpy as np

def make_linearly_separable():
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1,
                               n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)

a = [make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_linearly_separable()]
for b in a:
    X, y = b
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state = 0)


'''
names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
srhl = SimpleRandomHiddenLayer(n_hidden=10, activation_func='tanh', random_state=0)
classifier = [ELMClassifier(srhl)]
for clf in zip(names, classifier):
    print clf
    print '--'
'''


#names += 2
#print names