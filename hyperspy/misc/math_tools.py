import math

import numpy as np


def symmetrize(a):
    return a + a.swapaxes(0, 1) - np.diag(a.diagonal())


def antisymmetrize(a):
    return a - a.swapaxes(0, 1) + np.diag(a.diagonal())


def closest_nice_number(number):
    oom = 10 ** math.floor(math.log10(number))
    return oom * (number // oom)


def get_linear_interpolation(p1, p2, x):
    """Given two points in 2D returns y for a given x for y = ax + b

    Parameters
    ----------
    p1,p2 : (x, y)
    x : float

    Returns
    -------
    y : float

    """
    x1, y1 = p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    y = a * x + b
    return y


def order_of_magnitude(number):
    """Order of magnitude of the given number

    Parameters
    ----------
    number : float

    Returns
    -------
    Float
    """
    return math.floor(math.log10(number))


def isfloat(number):
    """Check if a number or array is of float type.

    This is necessary because e.g. isinstance(np.float32(2), float) is False.

    """
    if hasattr(number, "dtype"):
        return np.issubdtype(number, np.float)
    else:
        return isinstance(number, float)
