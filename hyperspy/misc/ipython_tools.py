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

import __main__
from distutils.version import LooseVersion
import os
from time import strftime


def get_ipython():
    """Get the global InteractiveShell instance.

    Returns None if no InteractiveShell instance is registered.
    """
    if is_it_running_from_ipython is False:
        return None
    import IPython
    ipy_version = LooseVersion(IPython.__version__)
    if ipy_version < LooseVersion("0.11"):
        from IPython import ipapi
        ip = ipapi.get()
    elif ipy_version < LooseVersion("1.0"):
        from IPython.core import ipapi
        ip = ipapi.get()
    else:
        ip = IPython.get_ipython()
    return ip


def is_it_running_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_interactive_ns():
    ip = get_ipython()
    if ip is None:
        return __main__.__dict__
    else:
        return ip.user_ns


def turn_logging_on(verbose=1):
    ip = get_ipython()
    if ip is None:
        return
    import IPython
    ipy_version = LooseVersion(IPython.__version__)
    if ipy_version < LooseVersion("0.11"):
        if verbose == 1:
            print("Logging is not supported by this version of IPython")
        return
    elif ip.logger.log_active is True:
        if verbose == 1:
            print("Already logging to " + ip.logger.logfname)
        return

    filename = os.path.join(os.getcwd(), 'hyperspy_log.py')
    new = not os.path.exists(filename)
    ip.logger.logstart(logfname=filename, logmode='append')
    if new:
        ip.logger.log_write(
            "#!/usr/bin/env python \n"
            "# ============================\n"
            "# %s \n" % strftime('%Y-%m-%d') +
            "# %s \n" % strftime('%H:%M') +
            "# ============================\n")
    if verbose == 1:
        print("\nLogging is active")
        print("The log is stored in the hyperspy_log.py file"
              " in the current directory")


def turn_logging_off():
    ip = get_ipython()
    if ip is None:
        return
    import IPython
    ipy_version = LooseVersion(IPython.__version__)
    if ipy_version < LooseVersion("0.11"):
        print("Logging is not supported by this version of IPython")
        return
    elif ip.logger.log_active is False:
        return

    ip.logger.logstop()
    print("The logger is off")
