# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import __main__
from distutils.version import StrictVersion
import os
from time import strftime

import IPython

ipy_version = StrictVersion(IPython.__version__)
ipy_011 = StrictVersion('0.11')

if ipy_version < ipy_011:
    from IPython import ipapi
else:
    from IPython.core import ipapi
ip = ipapi.get()
if ip is None:
    # Ipython is not installed, using Python's namespace.
    # TODO: this does not work with IPython > 0.11
    interactive_ns = __main__.__dict__
else:
    interactive_ns = ip.user_ns


def turn_logging_on(verbose = 1):
    
    if ipy_version < ipy_011:
        if verbose == 1:
            print("Logging is not supported by this version of IPython")
        return
    if ip.logger.log_active is True:
        if verbose == 1:
            print "Already logging to " + ip.logger.logfname
        return
    
    filename = os.path.join(os.getcwd(), 'hyperspy_log.py')
    new = not os.path.exists(filename)
    ip.logger.logstart(logfname=filename,logmode='append')
    if new:
        ip.logger.log_write(
            "#!/usr/bin/env python \n"
            "# ============================\n"
            "# %s \n" % strftime('%Y-%m-%d') +
            "# %s \n" % strftime('%H:%M') +
            "# ============================\n" )
    if verbose == 1:
        print("\nLogging is active")
        print("The log is stored in the hyperspy_log.py file"
              " in the current directory")
          
def turn_logging_off():
    if ipy_version < ipy_011:
        print("Logging is not supported by this version of IPython")
        return
    if ip.logger.log_active is False:
        return
        
    ip.logger.logstop()
    print("The logger is off")

        
