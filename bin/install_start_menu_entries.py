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
try:
    # When this script is run from inside the bdist_wininst installer,
    # file_created() and directory_created() are additional builtin
    # functions which write lines to Python23\pywin32-install.log. This is
    # a list of actions for the uninstaller, the format is inspired by what
    # the Wise installer also creates.
    file_created()
    is_bdist_wininst = True

except NameError:
    is_bdist_wininst = False  # we know what it is not - but not what it is :)

    def file_created(file):
        pass

    def directory_created(directory):
        pass

    def get_root_hkey():
        try:
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           root_key_name, winreg.KEY_CREATE_SUB_KEY)
            return winreg.HKEY_LOCAL_MACHINE
        except OSError as details:
            # Either not exist, or no permissions to create subkey means
            # must be HKCU
            return winreg.HKEY_CURRENT_USER


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


try:
    create_shortcut()
except NameError:
    # Create a function with the same signature as create_shortcut provided
    # by bdist_wininst
    def create_shortcut(path, description, filename,
                        arguments="", workdir="", iconpath="", iconindex=0):
        import pythoncom
        from win32com.shell import shell, shellcon

        ilink = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                                           pythoncom.CLSCTX_INPROC_SERVER,
                                           shell.IID_IShellLink)
        ilink.SetPath(path)
        ilink.SetDescription(description)
        if arguments:
            ilink.SetArguments(arguments)
        if workdir:
            ilink.SetWorkingDirectory(workdir)
        if iconpath or iconindex:
            ilink.SetIconLocation(iconpath, iconindex)
        # now save it.
        ipf = ilink.QueryInterface(pythoncom.IID_IPersistFile)
        ipf.Save(filename, 0)


def create_weblink(
        address, link_name, hspy_sm_path, description, icon_path=None):
    # documentation
    link = os.path.join(hspy_sm_path, link_name)
    if os.path.isfile(link):
        os.remove(link)  # we want to make a new one
    create_shortcut(address, description, link, '', '', icon_path)
    file_created(link)


def admin_rights():
    return shell.IsUserAnAdmin()


def install_start_menu_entries():
    import hyperspy

    commons_sm = get_special_folder_path("CSIDL_COMMON_STARTMENU")
    local_sm = get_special_folder_path("CSIDL_STARTMENU")
    if admin_rights() is True:
        start_menu = commons_sm
    else:
        start_menu = local_sm
    hyperspy_install_path = os.path.dirname(hyperspy.__file__)
    logo_path = os.path.expandvars(os.path.join(hyperspy_install_path,
                                                'data'))
    hyperspy_qtconsole_bat = os.path.join(sys.prefix,
                                          'Scripts',
                                          'hyperspy_qtconsole.bat')
    hyperspy_notebook_bat = os.path.join(sys.prefix,
                                         'Scripts',
                                         'hyperspy_notebook.bat')
    # Create the start_menu entry
    if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
        hspy_sm_path = os.path.join(start_menu, "Programs", "HyperSpy")
    else:
        hspy_sm_path = os.path.join(start_menu, "Programs", "HyperSpy")
    if os.path.isdir(hspy_sm_path):
        try:
            shutil.rmtree(hspy_sm_path)
        except:
            # Sometimes we get a permission error
            pass
    os.mkdir(hspy_sm_path)
    directory_created(hspy_sm_path)
    qtconsole_link_path = os.path.join(hspy_sm_path,
                                       'HyperSpy QtConsole.lnk')
    notebook_link_path = os.path.join(hspy_sm_path,
                                      'HyperSpy Notebook.lnk')
    create_shortcut(hyperspy_qtconsole_bat,
                    'HyperSpy QtConsole',
                    qtconsole_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_qtconsole_logo.ico'))
    print "Installed HyperSpy (QtConsole) shortcut"
    create_shortcut(hyperspy_notebook_bat,
                    'HyperSpy Notebook',
                    notebook_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_notebook_logo.ico'))
    print "Installed HyperSpy (Notebook) shortcut"
    file_created(qtconsole_link_path)
    file_created(notebook_link_path)

    links = [
        {
            'address': r"http://hyperspy.org/hyperspy-doc/current/index.html",
            'link_name': "HyperSpy Documentation.lnk",
            'hspy_sm_path': hspy_sm_path,
            'description': 'HyperSpy online documentation',
            'icon_path': os.path.join(logo_path, 'hyperspy_doc_logo.ico')},
        {
            'address': r"http://hyperspy.org",
            'link_name': "HyperSpy Homepage.lnk",
            'hspy_sm_path': hspy_sm_path,
            'description': 'HyperSpy homepage',
            'icon_path': os.path.join(logo_path, 'hyperspy_home_logo.ico')},
    ]
    for link in links:
        create_weblink(**link)
    print "Installed HyperSpy web links"

if __name__ == "__main__":
    if admin_rights() is True:
        install_start_menu_entries()
        print "All Start Menu entries were installed correctly"
    else:
        print("To add start menu entries, run this script "
              "with administrator rights")
