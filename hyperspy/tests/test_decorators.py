# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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


import warnings

import pytest

from hyperspy.decorators import deprecated, deprecated_argument
from hyperspy.exceptions import VisibleDeprecationWarning


class TestDeprecationWarning:
    def test_deprecation_since(self):
        """
        Ensure functions decorated with the custom deprecated
        decorator returns desired output, raises a desired warning, and
        gets the desired additions to their docstring.
        """

        @deprecated(since=0.7, alternative="bar", removal=0.8)
        def foo(n):
            """Some docstring."""
            return n + 1

        with pytest.warns(VisibleDeprecationWarning) as record:
            assert foo(4) == 5
        desired_msg = (
            "Function `foo()` is deprecated and will be removed in version 0.8. Use "
            "`bar()` instead."
        )
        assert str(record[0].message) == desired_msg
        assert foo.__doc__ == (
            "[*Deprecated*] Some docstring.\n\n"
            "Notes\n"
            "-----\n"
            ".. deprecated:: 0.7\n"
            f"   {desired_msg}"
        )  # fmt: skip

        @deprecated(since=1.9)
        def foo2(n):
            """
            Another docstring.

            Notes
            -----
            Some existing notes.
            """
            return n + 2

        with pytest.warns(VisibleDeprecationWarning) as record:
            assert foo2(4) == 6
        desired_msg2 = "Function `foo2()` is deprecated."
        assert str(record[0].message) == desired_msg2
        assert foo2.__doc__ == (
            "[*Deprecated*] Another docstring.\n\n"
            "Notes\n"
            "-----\n"
            "Some existing notes.\n\n"
            ".. deprecated:: 1.9\n"
            f"   {desired_msg2}"
        )  # fmt: skip

    def test_deprecation_no_old_doc(self):
        @deprecated(since=0.7, alternative="bar", removal=0.8)
        def foo(n):
            return n + 1

        with pytest.warns(VisibleDeprecationWarning) as record:
            assert foo(4) == 5
        desired_msg = (
            "Function `foo()` is deprecated and will be removed in version 0.8. Use "
            "`bar()` instead."
        )
        assert str(record[0].message) == desired_msg
        assert foo.__doc__ == (
            "[*Deprecated*] \n\n"
            "Notes\n"
            "-----\n"
            ".. deprecated:: 0.7\n"
            f"   {desired_msg}"
        )  # fmt: skip


class TestDeprecateArgument:
    def test_deprecate_argument(self):
        """
        Functions decorated with the custom `deprecated_argument`
        decorator returns desired output and raises a desired warning
        only if the argument is passed.
        """

        class Foo:
            @deprecated_argument(name="a", since="1.3", removal="1.4")
            def bar_arg(self, **kwargs):
                return kwargs

            @deprecated_argument(name="a", since="1.3", removal="1.4", alternative="b")
            def bar_arg_alt(self, **kwargs):
                return kwargs

        my_foo = Foo()

        # Does not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert my_foo.bar_arg(b=1) == {"b": 1}

        # Warns
        with pytest.warns(VisibleDeprecationWarning) as record1:
            assert my_foo.bar_arg(a=2) == {"a": 2}
        assert str(record1[0].message) == (
            "Argument `a` is deprecated and will be removed in version 1.4. "
            "To avoid this warning, please do not use `a`. See the documentation of "
            "`bar_arg()` for more details."
        )

        # Warns with alternative
        with pytest.warns(VisibleDeprecationWarning) as record2:
            assert my_foo.bar_arg_alt(a=3) == {"b": 3}
        assert str(record2[0].message) == (
            "Argument `a` is deprecated and will be removed in version 1.4. "
            "To avoid this warning, please do not use `a`. Use `b` instead. See the "
            "documentation of `bar_arg_alt()` for more details."
        )
