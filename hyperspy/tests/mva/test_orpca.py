import numpy as np
import scipy.linalg

import nose.tools as nt
from nose.plugins.skip import SkipTest

from hyperspy.learn.rpca import orpca

class TestORPCA:
    def setUp(self):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 1024 # Number of samples
        r = 5
        s = 0.01

        # Low-rank and sparse error matrices
        rng = np.random.RandomState(123)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = np.dot(U, V.T)
        E = 100 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.lambda1 = 1.0 / np.sqrt(n)
        self.lambda2 = 1.0 / np.sqrt(n)
        self.U = U
        self.A = A
        self.E = E
        self.X = X

        # Test tolerance
        self.tol = 1e-3

    def test_default(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A) / (self.m * self.n)
        nt.assert_true(normX < self.tol)

        # Check the low-rank component rank
        rankX = np.linalg.matrix_rank(X, self.tol)
        np.testing.assert_almost_equal(rankX, self.rank)

    def test_mask(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank,
                                 mask=self.E)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A) / (self.m * self.n)
        nt.assert_true(normX < self.tol)

        # Check the low-rank component rank
        rankX = np.linalg.matrix_rank(X, self.tol)
        np.testing.assert_almost_equal(rankX, self.rank)

    def test_method(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, method='BCD')

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A) / (self.m * self.n)
        nt.assert_true(normX < self.tol)

        # Check the low-rank component rank
        rankX = np.linalg.matrix_rank(X, self.tol)
        np.testing.assert_almost_equal(rankX, self.rank)

    def test_regularization(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank,
                                 lambda1=self.lambda1, lambda2=self.lambda2)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A) / (self.m * self.n)
        nt.assert_true(normX < self.tol)

        # Check the low-rank component rank
        rankX = np.linalg.matrix_rank(X, self.tol)
        np.testing.assert_almost_equal(rankX, self.rank)
