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
import _winreg
import win32api
from win32com.shell import shell
import shutil

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
        except OSError, details:
            # Either not exist, or no permissions to create subkey means
            # must be HKCU
            return winreg.HKEY_CURRENT_USER


def get_special_folder_path(path_name):
    import pythoncom
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


def uninstall_hyperspy_here():
    for env in ('qtconsole', 'notebook'):
        try:
            if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
                _winreg.DeleteKey(
                    _winreg.HKEY_LOCAL_MACHINE,
                    r'Software\Classes\Folder\Shell\HyperSpy_%s_here\Command' %
                    env)
                _winreg.DeleteKey(
                    _winreg.HKEY_LOCAL_MACHINE,
                    r'Software\Classes\Folder\Shell\HyperSpy_%s_here' %
                    env)
            else:  # Vista or newer
                _winreg.DeleteKey(
                    _winreg.HKEY_CLASSES_ROOT,
                    r'Directory\shell\hyperspy_%s_here\Command' %
                    env)
                _winreg.DeleteKey(
                    _winreg.HKEY_CLASSES_ROOT,
                    r'Directory\shell\hyperspy_%s_here' %
                    env)
                _winreg.DeleteKey(
                    _winreg.HKEY_CLASSES_ROOT,
                    r'Directory\Background\shell\hyperspy_%s_here\Command' %
                    env)
                _winreg.DeleteKey(
                    _winreg.HKEY_CLASSES_ROOT,
                    r'Directory\Background\shell\hyperspy_%s_here' %
                    env)
            print("HyperSpy %s here correctly uninstalled" % env)
        except:
            print("Failed to uninstall HyperSpy %s here" % env)


def install_hyperspy_here(hspy_qtconsole_logo_path, hspy_notebook_logo_path):
    # First uninstall old HyperSpy context menu entries
    try:
        if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
            _winreg.DeleteKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_here')
        else:  # Vista or newer
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_here')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_here\Command')
            _winreg.DeleteKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_here')
        uninstall_hyperspy_here()
    except:
        # The old entries were not present, so we do nothing
        pass

    # Install the context menu entries for the qtconsole and the IPython
    # notebook
    logos = {'qtconsole': hspy_qtconsole_logo_path, 'notebook': hspy_notebook_logo_path}
    for env in ('qtconsole', 'notebook'):
        script = os.path.join(sys.prefix, 'Scripts', "hyperspy_%s.bat" % env)
        if sys.getwindowsversion()[0] < 6.:  # Before Windows Vista
            key = _winreg.CreateKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s here" %
                env)
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_LOCAL_MACHINE,
                r'Software\Classes\Folder\Shell\HyperSpy_%s_here\Command' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_EXPAND_SZ,
                script +
                " \"%L\"")
            key.Close()
        else:  # Windows Vista and above
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s here" %
                env)
            _winreg.SetValueEx(
                key,
                'Icon',
                0,
                _winreg.REG_SZ,
                logos[env]
            )
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\shell\hyperspy_%s_here\Command' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_EXPAND_SZ,
                script +
                " \"%L\"")
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_%s_here' %
                env)
            _winreg.SetValueEx(
                key,
                "",
                0,
                _winreg.REG_SZ,
                "HyperSpy %s Here" %
                env)
            _winreg.SetValueEx(
                key,
                'Icon',
                0,
                _winreg.REG_SZ,
                logos[env]
            )
            key.Close()
            key = _winreg.CreateKey(
                _winreg.HKEY_CLASSES_ROOT,
                r'Directory\Background\shell\hyperspy_%s_here\Command' %
                env)
            _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, script)
            key.Close()

    print("HyperSpy here correctly installed")


def uninstall():
    import hyperspy
    import shutil

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
        hspy_sm_path = os.path.join(start_menu, "HyperSpy")
    if os.path.isdir(hspy_sm_path):
        try:
            shutil.rmtree(hspy_sm_path)
        except:
            # Sometimes we get a permission error
            pass

    # remove the shortcuts directory
    if sys.getwindowsversion()[0] < 6.:  # Older than Windows Vista:
        hspy_sm_path = os.path.join(start_menu, "Programs", "HyperSpy")
        if os.path.isdir(hspy_sm_path):
            try:
                shutil.rmtree(hspy_sm_path)
            except:
                # Sometimes we get a permission error
                pass
    else:
        hspy_sm_path = os.path.join(start_menu, "HyperSpy")
        if os.path.isdir(hspy_sm_path):
            try:
                shutil.rmtree(hspy_sm_path)
            except:
                # Sometimes we get a permission error
                pass

    if admin_rights() is True:
        uninstall_hyperspy_here()
    else:
        print("To correctly remove shortcuts and context menu \n"
              "entries, rerun with administrator rights")


def install():
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
    hspy_qt_logo_path = os.path.join(logo_path,
                                     'hyperspy_qtconsole_logo.ico')
    hspy_nb_logo_path = os.path.join(logo_path,
                                     'hyperspy_notebook_logo.ico')
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
        hspy_sm_path = os.path.join(start_menu, "HyperSpy")
    if os.path.isdir(hspy_sm_path):
        try:
            shutil.rmtree(hspy_sm_path)
        except:
            # Sometimes we get a permission error
            pass
    os.mkdir(hspy_sm_path)
    directory_created(hspy_sm_path)
    qtconsole_link_path = os.path.join(hspy_sm_path,
                                       'hyperspy_qtconsole.lnk')
    notebook_link_path = os.path.join(hspy_sm_path,
                                      'hyperspy_notebook.lnk')
    create_shortcut(hyperspy_qtconsole_bat,
                    'HyperSpy QtConsole',
                    qtconsole_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_qtconsole_logo.ico'))
    create_shortcut(hyperspy_notebook_bat,
                    'HyperSpy Notebook',
                    notebook_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_notebook_logo.ico'))
    file_created(qtconsole_link_path)
    file_created(notebook_link_path)

    links = [
        {
            'address': r"http://hyperspy.org/hyperspy-doc/current/index.html",
            'link_name': "hyperspy_doc.lnk",
            'hspy_sm_path': hspy_sm_path,
            'description': 'HyperSpy online documentation',
            'icon_path': os.path.join(logo_path, 'hyperspy_doc_logo.ico')},
        {
            'address': r"http://hyperspy.org",
            'link_name': "hyperspy_homepage.lnk",
            'hspy_sm_path': hspy_sm_path,
            'description': 'HyperSpy homepage',
            'icon_path': os.path.join(logo_path, 'hyperspy_home_logo.ico')},
    ]
    for link in links:
        create_weblink(**link)

    if admin_rights() is True:
        install_hyperspy_here(hspy_qt_logo_path, hspy_nb_logo_path)
    else:
        print("To start HyperSpy from the context menu install HyperSpy "
              "with administrator rights")

    print "All was installed correctly"


if len(sys.argv) == 2:
    if sys.argv[1] == '-install':
        install()
    elif sys.argv[1] == '-uninstall':
        uninstall()
    else:
        print 'Please call function with one argument, either \'-install\' or \'-uninstall\'.'
else:
    print 'Please call function with one argument, either \'-install\' or \'-uninstall\'.'
