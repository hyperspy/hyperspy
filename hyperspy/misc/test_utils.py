# -*- coding: utf-8 -*-
"""
"""

from contextlib import contextmanager
import warnings
import nose.tools as nt


@contextmanager
def assert_warns(message=None, category=None):
    """
    Creates a context where a warning is expected to be given.

    Use in a `with` statement to enter the context. The warning should be
    raised before the context exits.

    If `message` is given, it checks that the passed string is a part of the
    warnings message (~`message in warning.message`).
    If `category` is given, it checks that the warning category is a subclass
    of the passed `category`.

    Examples:
    ---------
        >>> with assert_warns("This is a warning", UserWarning):
        ...     function_that_should_warn()
        ... # Code continues here
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield
        if len(w) == 0:
            raise AssertionError("No warnings were given!")
        elif len(w) > 1:
            raise AssertionError("Multiple warnings received: %s" % str(w))

        if category is not None:
            if not issubclass(w[0].category, category):
                raise AssertionError(
                    "Warning of type %s received, expected %s" %
                    (w[0].category, category))
        if message is not None:
            nt.assert_in(message, w[0].message[0])


@contextmanager
def ignore_warning(message="", category=None):
    with warnings.catch_warnings():
        if category:
            warnings.filterwarnings('ignore', message, category=category)
        else:
            warnings.filterwarnings('ignore', message)
        yield
