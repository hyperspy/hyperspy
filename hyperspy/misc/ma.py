import numpy as np
import dask.array as da
from numpy.ma.core import MaskedConstant

masked = MaskedConstant()

def asarray(signal, dtype=None):
    if signal._lazy:
        # changing datatype not allowed
        signal.data = da.ma.asanyarray(signal.data)
    else:
        signal.data = np.ma.asanyarray(signal.data, dtype=dtype)

def masked_equal(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_equal(signal.data, value)
    else:
        signal.data = np.ma.masked_equal(signal.data, value=value, copy=copy)


def masked_greater(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_greater(signal.data, value, copy=copy)
    else:
        signal.data = np.ma.masked_greater(signal.data, value=value, copy=copy)


def masked_greater_equal(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_greater_equal(signal.data, value)
    else:
        signal.data = np.ma.masked_greater_equal(signal.data, value=value, copy=copy)

def masked_inside(signal, v1, v2, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_inside(signal.data, v1, v2)
    else:
        signal.data = np.ma.masked_inside(signal.data, v1, v2, copy=copy)


def masked_invalid(signal, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_invalid(signal.data)
    else:
        signal.data = np.ma.masked_invalid(signal.data,copy=copy)


def masked_less(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_less(signal.data, value=value)
    else:
        signal.data = np.ma.masked_less(signal.data, value=value, copy=copy)


def masked_less_equal(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_less_equal(signal.data, value=value)
    else:
        signal.data = np.ma.masked_less_equal(signal.data, value=value, copy=copy)


def masked_not_equal(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_not_equal(signal.data, value=value)
    else:
        signal.data = np.ma.masked_not_equal(signal.data, value=value, copy=copy)


def masked_outside(signal, v1, v2, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_outside(signal.data, v1, v2)
    else:
        signal.data = np.ma.masked_outside(signal.data, v1, v2, copy=copy)


def masked_values(signal, value, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_values(signal.data, value)
    else:
        signal.data = np.ma.masked_values(signal.data, value, copy=copy)


def masked_where(condition, signal, copy=True):
    if signal._lazy:
        signal.data = da.ma.masked_where(condition, signal.data)
    else:
        signal.data = np.ma.masked_where(signal.data, condition, copy=copy)
