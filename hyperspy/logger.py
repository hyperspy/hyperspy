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

import logging
import sys


def set_log_level(level):
    """
    Convenience function to set the log level of all hyperspy modules.

    Note: The log level of all other modules are left untouched.

    Parameters
    ----------
    level : int or str
        The log level to set. Any values that `logging.Logger.setLevel()`
        accepts are valid. The default options are:

        - 'CRITICAL'
        - 'ERROR'
        - 'WARNING'
        - 'INFO'
        - 'DEBUG'
        - 'NOTSET'

    Examples
    --------
    For normal logging of hyperspy functions, you can set the log level like
    this:

    >>> import hyperspy.api as hs
    >>> hs.set_log_level('INFO')
    >>> hs.load('my_file.dm3') # doctest: +SKIP
    INFO:rsciio.digital_micrograph:DM version: 3
    INFO:rsciio.digital_micrograph:size 4796607 B
    INFO:rsciio.digital_micrograph:Is file Little endian? True
    INFO:rsciio.digital_micrograph:Total tags in root group: 15
    <Signal2D, title: My file, dimensions: (|1024, 1024)>

    If you need the log output during the initial import of hyperspy, you
    should set the log level like this:

    >>> from hyperspy.logger import set_log_level
    >>> hs.set_log_level('DEBUG')
    >>> import hyperspy.api as hs # doctest: +SKIP
    DEBUG:hyperspy.gui:Loading hyperspy.gui
    DEBUG:hyperspy.gui:Current MPL backend: TkAgg
    DEBUG:hyperspy.gui:Current ETS toolkit: qt4
    DEBUG:hyperspy.gui:Current ETS toolkit set to: null

    """
    logger = initialize_logger("hyperspy")
    logger.setLevel(level)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter.

    This class is used to format the log output. The colors can be changed
    by changing the ANSI escape codes in the class variables.

    This is a modified version of both this
    https://github.com/herzog0/best_python_logger
    and this https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/56944256#56944256
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green = "\x1b[1;32m"
    format = "%(levelname)s | Hyperspy | %(message)s (%(name)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def initialize_logger(*args):
    """Creates a pretty logging instance where the colors can be changed
    via the ColoredFormatter class.  Any arguments passed to initialize_logger
    will be passed to `logging.getLogger`

    The logging output will also be redirected from the standard error file to
    the standard output file.
    """
    formatter = ColoredFormatter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    _logger = logging.getLogger(*args)
    # Remove existing handler
    while len(_logger.handlers):
        _logger.removeHandler(_logger.handlers[0])
    _logger.addHandler(handler)
    return _logger
