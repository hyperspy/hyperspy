# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

class MountainsMapFileError(Exception):

    def __init__(self, msg = "Corrupt Mountainsmap file"):
        self.error =  msg

    def __str__(self):
        return repr(self.error)

class ByteOrderError(Exception):

    def __init__(self, order=''):
        self.byte_order = order

    def __str__(self):
        return repr(self.byte_order)


class DM3FileVersionError(Exception):

    def __init__(self, value=''):
        self.dm3_version = value

    def __str__(self):
        return repr(self.dm3_version)


class DM3TagError(Exception):

    def __init__(self, value=''):
        self.dm3_tag = value

    def __str__(self):
        return repr(self.dm3_tag)


class DM3DataTypeError(Exception):

    def __init__(self, value=''):
        self.dm3_dtype = value

    def __str__(self):
        return repr(self.dm3_dtype)


class DM3TagTypeError(Exception):

    def __init__(self, value=''):
        self.dm3_tagtype = value

    def __str__(self):
        return repr(self.dm3_tagtype)


class DM3TagIDError(Exception):

    def __init__(self, value=''):
        self.dm3_tagID = value

    def __str__(self):
        return repr(self.dm3_tagID)


class ImageIDError(Exception):

    def __init__(self, value=''):
        self.image_id = value

    def __str__(self):
        return repr(self.image_id)


class ImageModeError(Exception):

    def __init__(self, value=''):
        self.mode = value

    def __str__(self):
        return repr(self.mode)


class ShapeError(Exception):

    def __init__(self, value):
        self.error = value.shape

    def __str__(self):
        return repr(self.error)


class NoInteractiveError(Exception):

    def __init__(self):
        self.error = "HyperSpy must run in interactive mode to use this feature"

    def __str__(self):
        return repr(self.error)


class WrongObjectError(Exception):

    def __init__(self, is_str, must_be_str):
        self.error = ("A object of type %s was given, but a %s" % (
            is_str, must_be_str) + " object is required")

    def __str__(self):
        return repr(self.error)


class MissingParametersError(Exception):

    def __init__(self, parameters):
        par_str = ''
        for par in parameters:
            par_str += '%s,' % par
        self.error = "The following parameters are missing: %s" % par_str
        # Remove the last comma
        self.error = self.error[:-1]

    def __str__(self):
        return repr(self.error)


class DataDimensionError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class SignalDimensionError(Exception):

    def __init__(self, output_dimension, expected_output_dimension):
        self.output_dimension = output_dimension
        self.expected_output_dimension = expected_output_dimension
        self.msg = 'output dimension=%i, %i expected' % (
            self.output_dimension, self.expected_output_dimension)

    def __str__(self):
        return repr(self.msg)


class NavigationDimensionError(Exception):

    def __init__(self,
                 navigation_dimension,
                 expected_navigation_dimension):
        self.navigation_dimension = navigation_dimension
        self.expected_navigation_dimension = \
            expected_navigation_dimension
        self.msg = 'navigation dimension=%i, %s expected' % (
            self.navigation_dimension, self.expected_navigation_dimension)

    def __str__(self):
        return repr(self.msg)


class SignalSizeError(Exception):

    def __init__(self, signal_size, expected_signal_size):
        self.signal_size = signal_size
        self.expected_signal_size = expected_signal_size
        self.msg = 'signal_size=%i, %i expected' % (
            self.signal_size, self.expected_signal_size)

    def __str__(self):
        return repr(self.msg)


class NavigationSizeError(Exception):

    def __init__(self, navigation_size, expected_navigation_size):
        self.navigation_size = navigation_size
        self.expected_navigation_size = expected_navigation_size
        self.msg = 'navigation_size =%i, %i expected' % (
            self.navigation_size, self.expected_navigation_size)


class VisibleDeprecationWarning(UserWarning):

    """Visible deprecation warning.
    By default, python will not show deprecation warnings, so this class
    provides a visible one.

    """
    pass
