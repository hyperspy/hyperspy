#/usr/bin/python
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

import os
import sys
import _winreg
import win32api
from win32com.shell import shell
import shutil


def create_weblink(address, link_name, hspy_sm_path, description, icon_path=None):
    # documentation
    link = os.path.join(hspy_sm_path, link_name)
    if os.path.isfile(link):
        os.remove(link) #we want to make a new one
    create_shortcut(address,description, link,'','',icon_path)
    file_created(link)

def admin_rights():
    return shell.IsUserAnAdmin()

def uninstall_hyperspy_here():
    for env in ('qtconsole', 'notebook'):
        try:
            if sys.getwindowsversion()[0] < 6.: # Older than Windows Vista:
                _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here' % env)
            else: # Vista or newer
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here\Command' % env)
                _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here' % env)
            print("Hyperspy %s here correctly uninstalled" % env)
        except:
            print("Failed to uninstall Hyperspy %s here" % env)
            
    
def install_hyperspy_here():
    ## First uninstall old Hyperspy context menu entries
    try:
        if sys.getwindowsversion()[0] < 6.: # Older than Windows Vista:
            _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here\Command')
            _winreg.DeleteKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_here')
        else: # Vista or newer
            _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here\Command')
            _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_here')
            _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here\Command')
            _winreg.DeleteKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_here')
        uninstall_hyperspy_here()
    except:
        # The old entries were not present, so we do nothing
        pass

    ## Install the context menu entries for the qtconsole and the IPython notebook
    for env in ('qtconsole', 'notebook'):
        script = os.path.join(sys.prefix, 'Scripts', "hyperspy_%s.bat" % env)
        if sys.getwindowsversion()[0] < 6.: # Before Windows Vista
            key = _winreg.CreateKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here' % env)
            _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy %s here" % env)
            key.Close()
            key = _winreg.CreateKey(_winreg.HKEY_LOCAL_MACHINE, r'Software\Classes\Folder\Shell\Hyperspy_%s_here\Command'  % env)
            _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, script + " \"%L\"")
            key.Close()
        else: # Windows Vista and above
            key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here' % env)
            _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy %s here" % env)
            key.Close()
            key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\shell\hyperspy_%s_here\Command' % env)
            _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, script + " \"%L\"")
            key.Close()
            key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here' % env)
            _winreg.SetValueEx(key,"",0,_winreg.REG_SZ,"Hyperspy %s Here" % env)
            key.Close()
            key = _winreg.CreateKey(_winreg.HKEY_CLASSES_ROOT, r'Directory\Background\shell\hyperspy_%s_here\Command' % env)
            _winreg.SetValueEx(key, "", 0, _winreg.REG_EXPAND_SZ, script)
            key.Close()
    
    print("Hyperspy here correctly installed")

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
    hyperspy_qtconsole_bat = os.path.join(sys.prefix,
                                          'Scripts',
                                          'hyperspy_qtconsole.bat')
    hyperspy_notebook_bat = os.path.join(sys.prefix,
                                         'Scripts',
                                         'hyperspy_notebook.bat')
    # Create the start_menu entry
    if sys.getwindowsversion()[0] < 6.: # Older than Windows Vista:
        hspy_sm_path = os.path.join(start_menu, "Programs","Hyperspy")
    else:
        hspy_sm_path = os.path.join(start_menu, "Hyperspy")
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
                    'Hyperspy QtConsole',
                    qtconsole_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_qtconsole_logo.ico'))
    create_shortcut(hyperspy_notebook_bat,
                    'Hyperspy Notebook',
                    notebook_link_path,
                    "",
                    os.path.expanduser("~"),
                    os.path.join(logo_path,
                                 'hyperspy_notebook_logo.ico'))
    file_created(qtconsole_link_path)
    file_created(notebook_link_path)

    links = [
        {
        'address' : r"http://hyperspy.org/hyperspy-doc/current/index.html",
        'link_name' : "hyperspy_doc.lnk",
        'hspy_sm_path' : hspy_sm_path,
        'description' : 'Hyperspy online documentation',
        'icon_path' : os.path.join(logo_path, 'hyperspy_doc_logo.ico')},                 
       {
       'address' : r"http://hyperspy.org",
       'link_name' : "hyperspy_homepage.lnk",
       'hspy_sm_path' : hspy_sm_path,
       'description' : 'Hyperspy homepage',
       'icon_path' : os.path.join(logo_path, 'hyperspy_home_logo.ico')},
                ]
    for link in links:
        create_weblink(**link)

    if admin_rights() is True:
        install_hyperspy_here()
    else:
        print("To start Hyperspy from the context menu install Hyperspy "
              "with administrator rights")

    print "All was installed correctly"

if sys.argv[1] == '-install':
    install()
else:
    uninstall_hyperspy_here()
