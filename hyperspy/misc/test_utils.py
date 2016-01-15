# -*- coding: utf-8 -*-
"""
"""

from contextlib import contextmanager
import warnings
import nose.tools as nt


@contextmanager
def assert_warns(message=None, category=None):
    """
    Runs function f, and checks that it gives a warning.

    If `message` is given, it checks that the passed string is a part of the
    warnings message (~`message in warning.message`).
    If `category` is given, it checks that the warning category is a subclass
    of the passed `category`.
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
