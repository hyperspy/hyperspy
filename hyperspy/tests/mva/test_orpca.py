import numpy as np
import scipy.linalg

import nose.tools as nt
from nose.plugins.skip import SkipTest

from hyperspy.learn.rpca import orpca

def _ev(U, L, atol=1e-3):
    # Check the similarity between the original
    # subspace basis and the OR-PCA results using
    # the "Expressed Variance" metric.
    # Perfect recovery of subspace: ev = 1
    C = np.dot(U, U.T)
    numer = np.trace(np.dot(L.T, np.dot(C, L)))
    denom = np.trace(C)
    return (1. - (numer / denom) < atol)

class TestORPCA:
    def setUp(self):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 1024 # Number of samples
        r = 5
        s = 0.1

        # Low-rank and sparse error matrices
        rng = np.random.RandomState(123)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = np.dot(U, V.T)
        E = 1000 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.lambda1 = 1 / np.sqrt(m)
        self.lambda2 = 1 / np.sqrt(m)
        self.U = U
        self.A = A
        self.E = E
        self.X = X

        # Test tolerance
        self.tol = 1e-4

    def test_default(self):
        L, R, E, U, S, V = orpca(self.X, rank=self.rank)

        # Check the low-rank component
        normA = np.linalg.norm(np.dot(L, R) - self.A) / (self.m * self.n)
        nt.assert_true(normA < self.tol)

        # Check the error component
        normE = np.linalg.norm(E - self.E) / (self.m * self.n)
        nt.assert_true(normE < self.tol)


        # Check the expressed variance of the
        # recovered subspace
        nt.assert_true(_ev(self.U, L, self.tol))

    def test_mask(self):
        L, R, E, U, S, V = orpca(self.X, rank=self.rank,
                                 mask=self.E)

        # Check the low-rank component
        normA = np.linalg.norm(np.dot(L, R) - self.A) / (self.m * self.n)
        nt.assert_true(normA < self.tol)

        # Check the error component
        normE = np.linalg.norm(E - self.E) / (self.m * self.n)
        nt.assert_true(normE < self.tol)

        # Check the expressed variance of the
        # recovered subspace
        nt.assert_true(_ev(self.U, L, self.tol))

    def test_regularization(self):
        L, R, E, U, S, V = orpca(self.X, rank=self.rank,
                                 lambda1=self.lambda1, lambda2=self.lambda2)

        # Check the low-rank component
        normA = np.linalg.norm(np.dot(L, R) - self.A) / (self.m * self.n)
        nt.assert_true(normA < self.tol)

        # Check the error component
        normE = np.linalg.norm(E - self.E) / (self.m * self.n)
        nt.assert_true(normE < self.tol)

        # Check the expressed variance of the
        # recovered subspace
        nt.assert_true(_ev(self.U, L, self.tol))

    def test_method(self):
        L, R, E, U, S, V = orpca(self.X, rank=self.rank, method='BCD')

        #  Check the low-rank component
        normA = np.linalg.norm(np.dot(L, R) - self.A) / (self.m * self.n)
        nt.assert_true(normA < self.tol)

        # Check the error component
        normE = np.linalg.norm(E - self.E) / (self.m * self.n)
        nt.assert_true(normE < self.tol)

        # Check the expressed variance of the
        # recovered subspace
        nt.assert_true(_ev(self.U, L, self.tol))
