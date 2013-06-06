#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Stefano Mazzucco
#
# This file is part of dm3_data_plugin.
#
# dm3_data_plugin is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# dm3_data_plugin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

# utility functions

from __future__ import with_statement # for Python versions < 2.6

import copy
import re
import os

def fsdict(nodes, value, dictionary):
    """Populates the dictionary 'dic' in a file system-like
    fashion creating a dictionary of dictionaries from the
    items present in the list 'nodes' and assigning the value
    'value' to the innermost dictionary.
    
    'dic' will be of the type:
    dic['node1']['node2']['node3']...['nodeN'] = value
    where each node is like a directory that contains other
    directories (nodes) or files (values)
    """
    node = nodes.pop(0)
    if node not in dictionary:
        dictionary[node] = {}
    if len(nodes) != 0 and isinstance(dictionary[node], dict):
        fsdict(nodes,value, dictionary[node])
    else:
        dictionary[node] = value

        
class DictBrowser(object):
    """Dictionary Browser.

    This class adds browsing capabilities to dictionaries. That is very useful
    when dealing with big dictionaries of dictionaries.

    Declare an instance with e.g.:
    >>> db = DictBrowser(my_dictionary)

    Now you will be able to browse the contents of my_dictionary in a *nix
    fashion by:
    >>> db.ls(some.path)
    and
    >>> db.cd(some.path)

    note that the separator '.' (default) can be changed using the keyword sep
    when declaring the DictBrowser instance.

    See help(DictBrowser.ls) and help(DictBrowser.cd) for more information.
    
    """
    
    def __init__(self, dic={}, pwd=[], sep='.'):
        self.sep = sep
        self.home = dic
        self.dic = dic
        self.pwd = []
        self.cd(pwd) # update self.dic and self.pwd
        self.oldpwd = self.pwd[:]

    def __repr__(self):
        return self.dic.__repr__()
    
    def __str__(self):
        return self.dic.__str__()

    def __setitem__(self, indx, val):
        return self.dic.__setitem__(indx, val)

    def __getitem__(self, indx):
        return self.dic.__getitem__(indx)

    def ls(self, pwd=[], dbg=False):
        """List the contents of the instance's dictionary
        attribute 'dic' given the path in pwd in a *nix-like
        fashion.
    
        'pwd' can be either a list or a string of keys
        separated by the separator attribute 'sep' (defaults to '.')

        the special keyword pwd='..' lists the contents
        relative to the previous key (directory).

        if 'dbg' is True, useful information is printed on screen
        
        E.g.
        obj.ls('root.dir1.dir2.dir3')
        obj.ls(['root', 'dir1', 'dir2', 'dir3'])
        """
        pwd = pwd[:] # don't modify the input object, work with a copy

        if pwd == '..':
            dic = DictBrowser(dic=self.home, pwd=self.pwd[:-1])
            return dic.ls()
        
        if type(pwd) is str:
            pwd = pwd.split(self.sep) # turn pwd into a list
        try:
            cdir = pwd.pop(0)   # current directory
        except:
            cdir = ''
        if cdir:
            if pwd:
                try:
                    dic = DictBrowser(dic=self.dic[cdir])
                    return dic.ls(pwd)
                except KeyError, key:
                    if dbg:
                        print('Key %s does not exist. Nothing to do.'
                              % str(key))
                    return None
            else:
                try:
                    if type(self.dic[cdir]) is dict:
                        # 'sub-directory' (return content)
                        out = self.dic[cdir].keys()
                        out.sort()
                        return out
                    else:
                        # 'file' (return name (key) and value)
                        return cdir, self.dic[cdir]
                except KeyError, key:
                    if dbg:
                        print('Key %s does not exist. Nothing to do.'
                              % str(key))
                    return None
        else:
            try:
                out = self.dic.keys()
                out.sort()
                return out
            except:
                if dbg:
                    msg = 'An error occurred processing '
                    msg += 'the ls() method of '
                    msg += self.__class__.__name__
                    print(msg)
                return None

    def cd(self, pwd=[], dbg=False):
        """Updates the instance's 'dic' attribute to the
        sub-dictionary given by the path in 'pwd' in a
        *nix-like fashion.
        
        'dic' should be a dictionary of dictionaries
        
        'pwd' can be either a list or a string of keys
        separated by the separator attribute 'sep' (defaults to '.')

        'pwd' defaults to [], that is
        cd() brings you to the 'root' dictionary

        the special keyword pwd='..' updates 'dic' to
        the previous key (directory).

        the special keyword pwd='-' updates 'dic' to
        the old key (directory).

        if 'dbg' is True, useful information is printed on screen
        
        E.g.
        obj.cd('root.dir1.dir2.dir3')
        obj.cd(['root', 'dir1', 'dir2', 'dir3'])
        """

        pwd = pwd[:] # don't modify the input object, work with a copy

        if pwd == '..': # going to previous directory (in *nix: cd ..)
            self.oldpwd = self.pwd[:]
            self.pwd.pop()
            self.dic = self.home.copy()
            pwd = self.pwd[:]
            newdic = DictBrowser(dic=self.dic, pwd=pwd, sep=self.sep)
            self.dic = newdic.dic.copy() # update the 'dic' attribute
            self.pwd =  newdic.pwd[:]
        elif pwd == '-': # going to old directory (in *nix: cd -)
            self.dic = self.home.copy()
            pwd = self.oldpwd[:]
            self.oldpwd = self.pwd[:]
            newdic = DictBrowser(dic=self.dic, pwd=pwd, sep=self.sep)
            self.dic = newdic.dic.copy() # update the 'dic' attribute
            self.pwd =  newdic.pwd[:]
        else:
            if type(pwd) is str:
                pwd = pwd.split(self.sep) # turn pwd into a list
            try:
                cdir = pwd.pop(0) # current directory
            except:
                cdir = ''
            if cdir:
                try:
                    if type(self.dic[cdir]) is dict:
                        # 'sub-directory' (return content)
                        # print('entering', cdir) # DEBUG
                        self.dic = self.dic[cdir]
                        self.pwd.append(cdir)
                    else:
                        if dbg:
                            msg = 'Key "%s" ' % str(cdir)
                            msg += 'is not a (sub)dictionary.'
                            msg += ' Nothing to do.'
                            print(msg)                                  
                        return None
                    if pwd:
                        newdic = DictBrowser(dic=self.dic, pwd=pwd,
                                             sep=self.sep)
                        self.dic = newdic.dic.copy()
                        self.pwd += newdic.pwd
                except KeyError, key: # non existing key (directory)
                    if dbg:
                        msg = 'Key %s does not exist' % str(key)
                        msg += ' in current (sub)dictionary. Nothing to do.' 
                        print(msg)
                    return None
            else:
                self.dic = self.home.copy()
                self.oldpwd = self.pwd[:]
                self.pwd = []
                
    def interactive_browsing(self, path=''):
        """Interactively browse the contents of a path.

        The operation can be interrupted by typing Ctl-D (Unix) or
        Ctl-Z+Return (Windows)

        Parameters
        ----------
        path : string or list (optional)
               if not given, the current path (pwd) is explored

        """
        if type(path) is str:
            path = path.split(self.sep) # turn path into a list
        for i in xrange(len(path)):
            if path[i] == '':
                path.pop(i)
                
        contents = self.ls(path)
        
        if type(contents) is tuple:
            print(contents)
            print('done')
            return
        else:
            contents =  iter(contents)
            
        print("Starting interactive browsing, hit 'Return' to continue.")
        try:
            while not raw_input():
                try:
                    browse =  path + [contents.next(),]
                    print(browse)
                    print(self.ls(browse))
                except StopIteration:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
            
