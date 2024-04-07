# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import inspect
import logging
import warnings
from functools import wraps
from typing import Callable, Optional, Union

import numpy as np

_logger = logging.getLogger(__name__)


def lazify(func, **kwargs):
    from hyperspy.model import BaseModel
    from hyperspy.signal import BaseSignal

    @wraps(func)
    def lazified_func(self, *args, **kwds):
        for k in self.__dict__.keys():
            if not k.startswith("__"):
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
            if thing.startswith("test"):
                if not thing.startswith("test_lazy"):
                    newname = "test_lazy" + thing[4:]
                    if newname not in thelist:
                        newfunc = lazify(getattr(original_class, thing), **kwargs)
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
    from hyperspy.signal_tools import Signal1DRangeSelector
    from hyperspy.ui_registry import get_gui

    def wrapper(self, *args, **kwargs):
        if not args and not kwargs:
            range_selector = Signal1DRangeSelector(self)
            range_selector.on_close.append((cm, self))
            get_gui(range_selector, toolkey="hyperspy.interactive_range_selector")
        else:
            cm(self, *args, **kwargs)

    return wrapper


class deprecated:
    """Decorator to mark deprecated functions with an informative
    warning.

    Inspired by
    `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py>`_
    and `matplotlib
    <https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_api/deprecation.py>`_.
    """

    def __init__(
        self,
        since: Union[str, int, float],
        alternative: Optional[str] = None,
        alternative_is_function: bool = True,
        removal: Union[str, int, float, None] = None,
    ):
        """Visible deprecation warning.

        Parameters
        ----------
        since
            The release at which this API became deprecated.
        alternative
            An alternative API that the user may use in place of the
            deprecated API.
        alternative_is_function
            Whether the alternative is a function. Default is ``True``.
        removal
            The expected removal version.
        """
        self.since = since
        self.alternative = alternative
        self.alternative_is_function = alternative_is_function
        self.removal = removal

    def __call__(self, func: Callable):
        # Wrap function to raise warning when called, and add warning to
        # docstring
        if self.alternative is not None:
            if self.alternative_is_function:
                alt_msg = f" Use `{self.alternative}()` instead."
            else:
                alt_msg = f" Use `{self.alternative}` instead."
        else:
            alt_msg = ""
        if self.removal is not None:
            rm_msg = f" and will be removed in version {self.removal}"
        else:
            rm_msg = ""
        msg = f"Function `{func.__name__}()` is deprecated{rm_msg}.{alt_msg}"

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter(
                action="always", category=np.VisibleDeprecationWarning, append=True
            )
            func_code = func.__code__
            warnings.warn_explicit(
                message=msg,
                category=np.VisibleDeprecationWarning,
                filename=func_code.co_filename,
                lineno=func_code.co_firstlineno + 1,
            )
            return func(*args, **kwargs)

        # Modify docstring to display deprecation warning
        old_doc = inspect.cleandoc(func.__doc__ or "").strip("\n")
        notes_header = "\nNotes\n-----"
        new_doc = (
            f"[*Deprecated*] {old_doc}\n"
            f"{notes_header if notes_header not in old_doc else ''}\n"
            f".. deprecated:: {self.since}\n"
            f"   {msg.strip()}"  # Matplotlib uses three spaces
        )
        wrapped.__doc__ = new_doc

        return wrapped


class deprecated_argument:
    """Decorator to remove an argument from a function or method's
    signature.

    Adapted from `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py>`_.
    """

    def __init__(self, name, since, removal, alternative=None):
        self.name = name
        self.since = since
        self.removal = removal
        self.alternative = alternative

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if self.name in kwargs.keys():
                msg = (
                    f"Argument `{self.name}` is deprecated and will be removed in "
                    f"version {self.removal}. To avoid this warning, please do not use "
                    f"`{self.name}`. "
                )
                if self.alternative is not None:
                    msg += f"Use `{self.alternative}` instead. "
                    kwargs[self.alternative] = kwargs.pop(
                        self.name
                    )  # replace with alternative kwarg
                msg += f"See the documentation of `{func.__name__}()` for more details."
                warnings.simplefilter(
                    action="always", category=np.VisibleDeprecationWarning
                )
                func_code = func.__code__
                warnings.warn_explicit(
                    message=msg,
                    category=np.VisibleDeprecationWarning,
                    filename=func_code.co_filename,
                    lineno=func_code.co_firstlineno + 1,
                )
            return func(*args, **kwargs)

        return wrapped


def jit_ifnumba(*args, **kwargs):
    try:
        import numba

        if "nopython" not in kwargs:
            kwargs["nopython"] = True
        return numba.jit(*args, **kwargs)
    except ImportError:
        _logger.warning(
            "Numba is not installed, falling back to " "non-accelerated implementation."
        )

        def wrap1(func):
            def wrap2(*args2, **kwargs2):
                return func(*args2, **kwargs2)

            return wrap2

        return wrap1
