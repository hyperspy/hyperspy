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


import warnings

from hyperspy.misc.test_utils import ignore_warning, assert_warns, all_warnings


def warnsA():
    warnings.warn("Warning A!", UserWarning)


def warnsB():
    warnings.warn("Warning B!", DeprecationWarning)


def warnsC():
    warnings.warn("Warning C!")


def test_ignore_full_message():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(message="Warning A!"):
            warnsA()
        with ignore_warning(message="Warning B!"):
            warnsB()
        with ignore_warning(message="Warning C!"):
            warnsC()


def test_ignore_partial_message():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(message="Warning"):
            warnsA()
            warnsB()
            warnsC()


def test_ignore_regex_message():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(message="Warning .?!"):
            warnsA()
            warnsB()
            warnsC()


def test_ignore_message_fails():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(message="Warning [AB]!"):
            warnsA()
            warnsB()
            try:
                warnsC()
            except UserWarning as e:
                assert str(e) == "Warning C!"
            else:
                raise ValueError("Expected warning to give error!")
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(message="Warning A! Too much"):
            try:
                warnsA()
            except UserWarning as e:
                assert str(e) == "Warning A!"
            else:
                raise ValueError("Expected warning to give error!")


def test_ignore_type():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(category=UserWarning):
            warnsA()
            warnsC()
        with ignore_warning(category=DeprecationWarning):
            warnsB()


def test_ignore_type_fails():
    with all_warnings():
        warnings.simplefilter("error")
        with ignore_warning(category=UserWarning):
            try:
                warnsB()
            except DeprecationWarning as e:
                assert str(e) == "Warning B!"
            else:
                raise ValueError("Expected warning to give error!")


def test_assert_warns_full_message():
    with all_warnings():
        warnings.simplefilter("error")
        with assert_warns(message="Warning A!"):
            warnsA()
        with assert_warns(message="Warning B!"):
            warnsB()
        with assert_warns(message="Warning C!"):
            warnsC()

        with assert_warns(message=["Warning A!", "Warning B!", "Warning C!"]):
            warnsA()
            warnsB()
            warnsC()


def test_assert_warns_partial_message():
    with all_warnings():
        warnings.simplefilter("error")
        with assert_warns(message="Warning"):
            warnsA()
            warnsB()
            warnsC()


def test_assert_warns_regex_message():
    with all_warnings():
        warnings.simplefilter("error")
        with assert_warns(message="Warning .?!"):
            warnsA()
            warnsB()
            warnsC()


def test_assert_warns_message_fails():
    with all_warnings():
        warnings.simplefilter("error")
        try:
            with assert_warns(message="Warning [AB]!"):
                warnsC()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected!")
    with all_warnings():
        warnings.simplefilter("error")
        try:
            with assert_warns(message="Warning A! Too much"):
                warnsA()
        except ValueError:
            pass
        else:
            raise ValueError("ValueError expected!")


def test_assert_warns_type():
    with all_warnings():
        warnings.simplefilter("error")
        with assert_warns(category=UserWarning):
            warnsA()
            warnsC()
        with assert_warns(category=DeprecationWarning):
            warnsB()


def test_assert_warns_type_fails():
    with all_warnings():
        warnings.simplefilter("error")
        try:
            with assert_warns(category=UserWarning):
                warnsB()
        except ValueError:
            pass
        else:
            raise ValueError("Expected warning to give error!")
