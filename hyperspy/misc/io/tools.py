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


import os
import logging

_logger = logging.getLogger(__name__)


def dump_dictionary(file, dic, string='root', node_separator='.',
                    value_separator=' = '):
    for key in list(dic.keys()):
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(string + node_separator + key + value_separator +
                       str(dic[key]) + '\n')


def append2pathname(filename, to_append):
    """Append a string to a path name

    Parameters
    ----------
    filename : str
    to_append : str

    """
    pathname, extension = os.path.splitext(filename)
    return pathname + to_append + extension


def incremental_filename(filename, i=1):
    """If a file with the same file name exists, returns a new filename that
    does not exists.

    The new file name is created by appending `-n` (where `n` is an integer)
    to path name

    Parameters
    ----------
    filename : str
    i : int
       The number to be appended.
    """

    if os.path.isfile(filename):
        new_filename = append2pathname(filename, '-%s' % i)
        if os.path.isfile(new_filename):
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename


def ensure_directory(path):
    """Check if the path exists and if it does not create the directory"""
    directory = os.path.split(path)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def overwrite(fname):
    """ If file exists 'fname', ask for overwriting and return True or False,
    else return True.

    """
    if os.path.isfile(fname):
        message = "Overwrite '%s' (y/n)?\n" % fname
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != 'y') and (answer != 'n'):
                print('Please answer y or n.')
                answer = input(message)
            if answer.lower() == 'y':
                return True
            elif answer.lower() == 'n':
                return False
        except:
            # We are running in the IPython notebook that does not
            # support raw_input
            _logger.info("Your terminal does not support raw input. "
                         "Not overwriting. "
                         "To overwrite the file use `overwrite=True`")
            return False
    else:
        return True
