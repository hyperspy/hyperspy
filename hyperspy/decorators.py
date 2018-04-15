# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

# custom exceptions
from functools import wraps


def lazify(func, **kwargs):
    from hyperspy.signal import BaseSignal
    from hyperspy.model import BaseModel

    @wraps(func)
    def lazified_func(self, *args, **kwds):
        for k in self.__dict__.keys():
            if not k.startswith('__'):
                v = getattr(self, k)
                if isinstance(v, BaseSignal):
                    v = v.as_lazy()
                    setattr(self, k, v)
                elif isinstance(v, BaseModel):
                    if hasattr(v, "signal"):
                        am = v.signal.axes_manager
                        v.signal = v.signal.as_lazy()
                        # Keep the axes_manager from the original signal that
                        # the model assigns to the components
                        v.signal.axes_manager = am
        self.__dict__.update(kwargs)
        return func(self, *args, **kwds)
    return lazified_func


def lazifyTestClass(*args, **kwargs):
    def lazifyTest(original_class):
        original_class.lazify = lazify
        thelist = [k for k in original_class.__dict__.keys()]
        for thing in thelist:
            if thing.startswith('test'):
                if not thing.startswith('test_lazy'):
                    newname = 'test_lazy' + thing[4:]
                    if newname not in thelist:
                        newfunc = lazify(getattr(original_class, thing),
                                         **kwargs)
                        newfunc.__name__ = newname
                        setattr(original_class, newname, newfunc)

        return original_class
    if len(args):
        return lazifyTest(*args)
    else:
        return lazifyTest


def simple_decorator(decorator):
    """This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied.

    This decorator was taken from:
    http://wiki.python.org/moin/PythonDecoratorLibrary"""
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


@simple_decorator
def interactive_range_selector(cm):
    from hyperspy.ui_registry import get_gui
    from hyperspy.signal_tools import Signal1DRangeSelector

    def wrapper(self, *args, **kwargs):
        if not args and not kwargs:
            range_selector = Signal1DRangeSelector(self)
            range_selector.on_close.append((cm, self))
            get_gui(range_selector, toolkey="interactive_range_selector")
        else:
            cm(self, *args, **kwargs)
    return wrapper


def jit_ifnumba(*args, **kwargs):
    try:
        import numba
        if "nopython" not in kwargs:
            kwargs["nopython"] = True
        return numba.jit(*args, **kwargs)
    except ImportError:
        def wrap1(func):
            def wrap2(*args2, **kwargs2):
                return func(*args2, **kwargs2)
            return wrap2
        return wrap1
