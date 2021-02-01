import importlib
from lazyasd import lazyobject

@lazyobject
def pint():
    return importlib.import_module('pint')

@lazyobject
def sympy():
    return importlib.import_module('sympy')

@lazyobject
def Lambdify():
    return importlib.import_module('sympy.utilities.lambdify')

@lazyobject
def skimage_feature():
    return importlib.import_module('skimage.feature')

@lazyobject
def skimage_correlation():
    try:
        # For scikit-image >= 0.17.0
        return importlib.import_module('skimage.registration._phase_cross_correlation')
    except ModuleNotFoundError:
        return importlib.import_module('skimage.feature.register_translation')

@lazyobject
def dask_array():
    return importlib.import_module('dask.array')

@lazyobject
def threaded():
    return importlib.import_module('dask.threaded')

@lazyobject
def dask_diag():
    return importlib.import_module('dask.diagnostics')

