import numpy as np


def orthomax(A, gamma=1, reltol=1.4901e-07, maxit=256):
    # ORTHOMAX Orthogonal rotation of FA or PCA loadings.
    # Taken from metpy

    d, m = A.shape
    B = np.copy(A)
    T = np.eye(m)

    converged = False
    if (0 <= gamma) & (gamma <= 1):
        while converged is False:
            # Use Lawley and Maxwell's fast version
            D = 0
            for k in range(1, maxit + 1):
                Dold = D
                tmp11 = np.sum(np.power(B, 2), axis=0)
                tmp1 = np.matrix(np.diag(np.array(tmp11).flatten()))
                tmp2 = gamma * B
                tmp3 = d * np.power(B, 3)
                L, D, M = np.linalg.svd(
                    np.dot(A.transpose(), tmp3 - np.dot(tmp2, tmp1)))
                T = np.dot(L, M)
                D = np.sum(np.diag(D))
                B = np.dot(A, T)
                if np.abs(D - Dold) / D < reltol:
                    converged = True
                    break
    else:
        # Use a sequence of bivariate rotations
        for iter in range(1, maxit + 1):
            maxTheta = 0
            for i in range(0, m - 1):
                for j in range(i, m):
                    Bi = B[:, i]
                    Bj = B[:, j]
                    u = np.multiply(Bi, Bi) - np.multiply(Bj, Bj)
                    v = 2 * np.multiply(Bi, Bj)
                    usum = u.sum()
                    vsum = v.sum()
                    numer = 2 * np.dot(u.transpose(), v) - \
                        2 * gamma * usum * vsum / d
                    denom = (np.dot(u.transpose(), u) -
                             np.dot(v.transpose(), v) -
                             gamma * (usum ** 2 - vsum ** 2) / d)
                    theta = np.arctan2(numer, denom) / 4
                    maxTheta = max(maxTheta, np.abs(theta))
                    Tij = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
                    B[:, [i, j]] = np.dot(B[:, [i, j]], Tij)
                    T[:, [i, j]] = np.dot(T[:, [i, j]], Tij)
            if maxTheta < reltol:
                converged = True
                break
    return B, T
