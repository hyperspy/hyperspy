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


import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from hyperspy.misc.utils import DictionaryTreeBrowser


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
    if Path(fname).is_file():
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
        dictree[et.tag] = et.text
        return
    else:
        dictree.add_node(et.tag)
        if et.attrib:
            dictree[et.tag].add_dictionary(et.attrib)
        for child in et:
            xml2dtb(child, dictree[et.tag])


def convert_xml_to_dict(xml_object):
    if isinstance(xml_object, str):
        xml_object = ET.fromstring(xml_object)
    op = DictionaryTreeBrowser()
    xml2dtb(xml_object, op)
    return op
