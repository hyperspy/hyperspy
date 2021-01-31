import importlib
from lazyasd import lazyobject

@lazyobject
def pint():
    return importlib.import_module('pint')
