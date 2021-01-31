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

