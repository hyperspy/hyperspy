# /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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
import sys
from win32com.shell import shell
import shutil


# Tricks to use the functions that are present in the bdist_wininst package
# Obtained from https://github.com/saltstack/salt-windows-install
def get_special_folder_path(path_name):
    from win32com.shell import shellcon

    for maybe in """
        CSIDL_COMMON_STARTMENU CSIDL_STARTMENU CSIDL_COMMON_APPDATA
        CSIDL_LOCAL_APPDATA CSIDL_APPDATA CSIDL_COMMON_DESKTOPDIRECTORY
        CSIDL_DESKTOPDIRECTORY CSIDL_COMMON_STARTUP CSIDL_STARTUP
        CSIDL_COMMON_PROGRAMS CSIDL_PROGRAMS CSIDL_PROGRAM_FILES_COMMON
        CSIDL_PROGRAM_FILES CSIDL_FONTS""".split():
        if maybe == path_name:
            csidl = getattr(shellcon, maybe)
            return shell.SHGetSpecialFolderPath(0, csidl, False)
    raise ValueError("%s is an unknown path ID" % (path_name,))


def admin_rights():
    return shell.IsUserAnAdmin()


def uninstall_start_menu_entries():
    commons_sm = get_special_folder_path("CSIDL_COMMON_STARTMENU")
    start_menu = commons_sm

    # Remove the start menu entry
    if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
        hspy_sm_path = os.path.join(start_menu, "Programs", "HyperSpy")
    else:
        hspy_sm_path = os.path.join(start_menu, "Programs", "HyperSpy")
    if os.path.isdir(hspy_sm_path):
        try:
            shutil.rmtree(hspy_sm_path)
            print "HyperSpy Start Menu entries uninstalled correctly"
        except:
            # Sometimes we get a permission error
            print "Something has gone wrong. Start menu entries were not removed."
            pass


if __name__ == "__main__":
    if admin_rights() is True:
        uninstall_start_menu_entries()
    else:
        print("To remove start menu entries, run this script "
              "with administrator rights")
