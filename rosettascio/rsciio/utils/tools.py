# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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


import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from collections import OrderedDict

import numpy as np
from box import Box


_logger = logging.getLogger(__name__)


def dump_dictionary(
    file, dic, string="root", node_separator=".", value_separator=" = "
):
    for key in list(dic.keys()):
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(
                string + node_separator + key + value_separator + str(dic[key]) + "\n"
            )


def append2pathname(filename, to_append):
    """Append a string to a path name

    Parameters
    ----------
    filename : str
    to_append : str

    """
    p = Path(filename)
    return Path(p.parent, p.stem + to_append, p.suffix)


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
    filename = Path(filename)

    if filename.is_file():
        new_filename = append2pathname(filename, "-{i}")
        if new_filename.is_file():
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename


def ensure_directory(path):
    """Check if the path exists and if it does not, creates the directory."""
    # If it's a file path, try the parent directory instead
    p = Path(path)
    p = p.parent if p.is_file() else p

    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _logger.debug(f"Directory {p} already exists. Doing nothing.")


def overwrite(fname):
    """ If file exists 'fname', ask for overwriting and return True or False,
    else return True.

    Parameters
    ----------
    fname : str or pathlib.Path
        File to check for overwriting.

    Returns
    -------
    bool : 
        Whether to overwrite file.

    """
    if Path(fname).is_file() or (Path(fname).is_dir() and
                                 os.path.splitext(fname)[1] == '.zspy'):
        message = f"Overwrite '{fname}' (y/n)?\n"
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False
        except:
            # We are running in the IPython notebook that does not
            # support raw_input
            _logger.info(
                "Your terminal does not support raw input. "
                "Not overwriting. "
                "To overwrite the file use `overwrite=True`"
            )
            return False
    else:
        return True


def xml2dtb(et, dictree):
    if et.text:
        dictree.set_item(et.tag, et.text)
        return
    else:
        dictree.add_node(et.tag)
        if et.attrib:
            dictree[et.tag].merge_update(et.attrib)
        for child in et:
            xml2dtb(child, dictree[et.tag])

class DTBox(Box):
    def add_node(self, path):
        keys = path.split(".")
        for key in keys:
            if self.get(key) is None:
                self[key] = {}
            self = self[key]
    def set_item(self, path, value):
        if self.get(path) is None:
            self.add_node(path)
        self[path] = value
    def has_item(self, path):
        return self.get(path) is not None

def convert_xml_to_dict(xml_object):
    if isinstance(xml_object, str):
        xml_object = ET.fromstring(xml_object)
    op = DTBox(box_dots=True)
    xml2dtb(xml_object, op)
    return op

def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dict
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    """
    if dictionary is None:
        dictionary = OrderedDict()
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 else sarray[name]
    return dictionary


def dict2sarray(dictionary, sarray=None, dtype=None):
    """Populates a struct array from a dictionary

    Parameters
    ----------
    dictionary: dict
    sarray: struct array or None
        Either sarray or dtype must be given. If sarray is given, it is
        populated from the dictionary.
    dtype: None, numpy dtype or dtype list
        If sarray is None, dtype must be given. If so, a new struct array
        is created according to the dtype, which is then populated.

    Returns
    -------
    Structure array

    """
    if sarray is None:
        if dtype is None:
            raise ValueError("Either sarray or dtype need to be specified.")
        sarray = np.zeros((1,), dtype=dtype)
    for name in set(sarray.dtype.names).intersection(set(dictionary.keys())):
        if len(sarray[name]) == 1:
            sarray[name][0] = dictionary[name]
        else:
            sarray[name] = dictionary[name]
    return sarray


