# -*- coding: utf-8 -*-
"""
"""

from contextlib import contextmanager
import warnings
import re

import numpy as np
import numpy.testing as nt
from numpy.testing import assert_allclose


@contextmanager
def ignore_warning(message="", category=None):
    with warnings.catch_warnings():
        if category:
            warnings.filterwarnings('ignore', message, category=category)
        else:
            warnings.filterwarnings('ignore', message)
        yield


# TODO: Remove _all_warnings when moved to Python > 3.4,
# ref http://bugs.python.org/issue4180

# Licence for code below:

# Copyright (C) 2011, the scikit-image team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name of skimage nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

@contextmanager
def all_warnings():
    """
    Context for use in testing to ensure that all warnings are raised.
    Examples
    --------
    >>> import warnings
    >>> def foo():
    ...     warnings.warn(RuntimeWarning("bar"))
    We raise the warning once, while the warning filter is set to "once".
    Hereafter, the warning is invisible, even with custom filters:
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('once')
    ...     foo()
    We can now run ``foo()`` without a warning being raised:
    >>> from numpy.testing import assert_warns
    >>> foo()
    To catch the warning, we call in the help of ``all_warnings``:
    >>> with all_warnings():
    ...     assert_warns(RuntimeWarning, foo)
    """

    # Whenever a warning is triggered, Python adds a __warningregistry__
    # member to the *calling* module.  The exercize here is to find
    # and eradicate all those breadcrumbs that were left lying around.
    #
    # We proceed by first searching all parent calling frames and explicitly
    # clearing their warning registries (necessary for the doctests above to
    # pass).  Then, we search for all submodules of skimage and clear theirs
    # as well (necessary for the skimage test suite to pass).

    # frame = inspect.currentframe()
    # if frame:
    #     for f in inspect.getouterframes(frame):
    #         f[0].f_locals['__warningregistry__'] = {}
    # del frame

    # for mod_name, mod in list(sys.modules.items()):
    #     if 'six.moves' in mod_name:
    #         continue
    #     try:
    #         mod.__warningregistry__.clear()
    #     except AttributeError:
    #         pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


@contextmanager
def assert_warns(message=None, category=None):
    """Context for use in testing to catch known warnings matching regexes

    Parameters
    ----------
    message : (list of) strings or compiled regexes
        Regexes for the desired warning to catch
    category : type or list of types
        Warning categories for the desired warning to catch
    Examples
    --------
    >>> from skimage import data, img_as_ubyte, img_as_float
    >>> with assert_warns(['precision loss']):
    ...     d = img_as_ubyte(img_as_float(data.coins()))
    Notes
    -----
    Upon exiting, it checks the recorded warnings for the desired matching
    pattern(s).
    Raises a ValueError if any match was not found or an unexpected
    warning was raised.
    Allows for three types of behaviors: "and", "or", and "optional" matches.
    This is done to accomodate different build enviroments or loop conditions
    that may produce different warnings.  The behaviors can be combined.
    If you pass multiple patterns, you get an orderless "and", where all of the
    warnings must be raised.
    If you use the "|" operator in a pattern, you can catch one of several warnings.
    Finally, you can use "|\A\Z" in a pattern to signify it as optional.
    """
    if isinstance(message, (str, re._pattern_type)):
        message = [message]
    elif message is None:
        message = tuple()
    with all_warnings() as w:
        # enter context
        yield w
        # exited user context, check the recorded warnings
        remaining = [m for m in message if '\A\Z' not in m.split('|')]
        for warn in w:
            found = False
            for match in message:
                if re.search(match, str(warn.message)) is not None:
                    found = True
                    if match in remaining:
                        remaining.remove(match)
            if category is not None:
                if not issubclass(w[0].category, category):
                    raise ValueError(
                        "Warning of type %s received, expected %s" %
                        (w[0].category, category))
                found = True
            if not found:
                raise ValueError('Unexpected warning: %s' % str(warn.message))
        if len(remaining) > 0:
            msg = 'No warning raised matching:\n%s' % '\n'.join(remaining)
            raise ValueError(msg)


# Adapted from:
# https://github.com/gem/oq-engine/blob/master/openquake/server/tests/helpers.py
def assert_deep_almost_equal(actual, expected, *args, **kwargs):
    """ Assert that two complex structures have almost equal contents.
    Compares lists, dicts and tuples recursively. Checks numeric values
    using :py:meth:`assert_allclose` and
    checks all other values with :py:meth:`assert_equal`.
    Accepts additional positional and keyword arguments and pass those
    intact to assert_allclose() (that's how you specify comparison
    precision).
    Parameters
    ----------
    actual: lists, dicts or tuples

    expected: lists, dicts or tuples
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex)):
            assert_allclose(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2,
                                         __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assert_deep_almost_equal(expected[key], actual[key],
                                         __trace=repr(key), *args, **kwargs)
        else:
            assert expected == actual
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc, trace))
        raise exc
