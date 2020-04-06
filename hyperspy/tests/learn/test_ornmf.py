import numpy as np
import pytest

from hyperspy.learn.ornmf import ornmf
from hyperspy.signals import Signal1D


def compare_norms(a, b, tol=5e-3):
    assert a.shape == b.shape

    m, n = a.shape
    tol *= m * n
    n1, n2 = list(map(np.linalg.norm, [a, b]))

    assert np.linalg.norm((a / n1) - (b / n2)) < tol


m = 128
n = 256
r = 3

rng = np.random.RandomState(101)
U = rng.uniform(0, 1, (m, r))
V = rng.uniform(0, 1, (n, r))
X = np.dot(U, V.T)
np.divide(X, max(1.0, np.linalg.norm(X)), out=X)

sparse = 0.05  # Fraction of corrupted pixels
E = 100 * rng.binomial(1, sparse, X.shape)
Y = X + E


@pytest.mark.parametrize("project", [True, False])
def test_default(project):
    W, H = ornmf(X, r, project=project)
    compare_norms(np.dot(W, H), X)

    assert W.shape == U.shape
    assert H.shape == V.T.shape


def test_batch_size():
    W, H = ornmf(X, r, batch_size=2)
    compare_norms(np.dot(W, H), X)

    assert W.shape == U.shape
    assert H.shape == V.T.shape


def test_store_error():
    Xhat, Ehat, W, H = ornmf(X, r, store_error=True)
    compare_norms(Xhat, X)

    assert Xhat.shape == X.shape
    assert Ehat.shape == E.shape


def test_corrupted_default():
    W, H = ornmf(Y, r)
    compare_norms(np.dot(W, H), X)


def test_robust():
    W, H = ornmf(X, r, method="RobustPGD")
    compare_norms(np.dot(W, H), X)


def test_corrupted_robust():
    W, H = ornmf(Y, r, method="RobustPGD")
    compare_norms(np.dot(W, H), X)


def test_no_method():
    with pytest.raises(ValueError, match=f"'method' not recognised"):
        W, H = ornmf(X, r, method="uniform")


def test_subspace_tracking():
    W, H = ornmf(X, r, method="MomentumSGD")
    compare_norms(np.dot(W, H), X)


@pytest.mark.parametrize("subspace_learning_rate", [None, 1.1])
def test_subspace_tracking_learning_rate(subspace_learning_rate):
    W, H = ornmf(
        X, r, method="MomentumSGD", subspace_learning_rate=subspace_learning_rate
    )
    compare_norms(np.dot(W, H), X)


@pytest.mark.parametrize("subspace_momentum", [None, 0.9])
def test_subspace_tracking_momentum(subspace_momentum):
    W, H = ornmf(X, r, method="MomentumSGD", subspace_momentum=subspace_momentum)
    compare_norms(np.dot(W, H), X)

    with pytest.raises(ValueError, match=f"must be a float between 0 and 1"):
        W, H = ornmf(X, r, method="MomentumSGD", subspace_momentum=1.9)


@pytest.mark.parametrize("poisson", [True, False])
def test_signal(poisson):
    # Note that s1.decomposition() operates on the transpose
    # i.e. (n_samples, n_features).
    x = Y.T.copy().reshape(16, 16, 128)

    if poisson:
        x -= x.min()
        x[x <= 0] = 1e-16

    s1 = Signal1D(x)

    X_out, E_out = s1.decomposition(
        normalize_poissonian_noise=poisson,
        algorithm="ornmf",
        output_dimension=r,
        return_info=True,
    )

    # Check the low-rank component MSE
    compare_norms(X_out, X.T)
