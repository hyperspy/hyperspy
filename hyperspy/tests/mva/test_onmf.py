import numpy as np
from hyperspy.learn.onmf import onmf

def compare(a, b):
    n1, n2 = list(map(np.linalg.norm, [a, b]))
    return np.linalg.norm((a/n1) - (b/n2))


m = 512
n = 256
r = 3
rng = np.random.RandomState(101)
U = rng.uniform(0, 1, (m, r))
V = rng.uniform(0, 1, (r, n))
X = np.dot(U, V)

sparse = 0.1 # Fraction of corrupted pixels
E = 1000 * rng.binomial(1, sparse, X.shape)
Y = X + E 

def test_default():
    W, H = onmf(X, r)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.06

def test_corrupted_default():
    W, H = onmf(Y, r)
    res =  compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.13

def test_robust():
    W, H = onmf(X, r, robust=True)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.05

def test_corrupted_robust():
    W, H = onmf(Y, r, robust=True)
    res =  compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.095
