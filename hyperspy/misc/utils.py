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

from __future__ import division
import inspect
import copy
import types
import re
from StringIO import StringIO
import codecs
import collections

import numpy as np


def generate_axis(origin, step, N, index=0):
    """Creates an axis given the origin, step and number of channels

    Alternatively, the index of the origin channel can be specified.

    Parameters
    ----------
    origin : float
    step : float
    N : number of channels
    index : int
        index of origin

    Returns
    -------
    Numpy array
    
    """
    return np.linspace(origin-index*step, origin+step*(N-1-index), N)



def unfold_if_multidim(signal):
    """Unfold the SI if it is 2D

    Parameters
    ----------
    signal : Signal instance

    Returns
    -------

    Boolean. True if the SI was unfolded by the function.
    
    """
    if len(signal.axes_manager._axes) > 2:
        print "Automatically unfolding the SI"
        signal.unfold()
        return True
    else:
        return False



def str2num(string, **kargs):
    """Transform a a table in string form into a numpy array

    Parameters
    ----------
    string : string

    Returns
    -------
    numpy array
    
    """
    stringIO = StringIO(string)
    return np.loadtxt(stringIO, **kargs)

    
_slugify_strip_re = re.compile(r'[^\w\s-]')
_slugify_hyphenate_re = re.compile(r'[-\s]+')
def slugify(value, valid_variable_name=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    
    Adapted from Django's "django/template/defaultfilters.py".
    
    """
    import unicodedata
    if not isinstance(value, unicode):
        value = value.decode('utf8')
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(_slugify_strip_re.sub('', value).strip())
    value = _slugify_hyphenate_re.sub('_', value)
    if valid_variable_name is True:
        if value[:1].isdigit():
            value = u'Number_' + value
    return value
    
class DictionaryBrowser(object):
    """A class to comfortably access some parameters as attributes
    
    """

    def __init__(self, dictionary={}):
        super(DictionaryBrowser, self).__init__()
        self._load_dictionary(dictionary)

    def _load_dictionary(self, dictionary):
        for key, value in dictionary.iteritems():
            self.__setattr__(key, value)
            
    def export(self, filename, encoding = 'utf8'):
        """Export the dictionary to a text file
        
        Parameters
        ----------
        filename : str
            The name of the file without the extension that is
            txt by default
        encoding : valid encoding str
        
        """
        f = codecs.open(filename, 'w', encoding = encoding)
        f.write(self._get_print_items(max_len=None))
        f.close()

    def _get_print_items(self, padding = '', max_len=20):
        """Prints only the attributes that are not methods
        
        """
        string = ''
        eoi = len(self.__dict__)
        j = 0
        for key_, value in iter(sorted(self.__dict__.iteritems())):
            if key_[:1] == "_":
                eoi -= 1
                continue
            if type(key_) != types.MethodType:
                key = ensure_unicode(value['key'])
                value = ensure_unicode(value['value'])
                if isinstance(value, DictionaryBrowser):
                    if j == eoi - 1:
                        symbol = u'└── '
                    else:
                        symbol = u'├── '
                    string += u'%s%s%s\n' % (padding, symbol, key)
                    if j == eoi - 1:
                        extra_padding = u'    '
                    else:
                        extra_padding = u'│   '
                    string += value._get_print_items(
                        padding + extra_padding)
                else:
                    if j == eoi - 1:
                        symbol = u'└── '
                    else:
                        symbol = u'├── '
                    strvalue = unicode(value)
                    if max_len is not None and \
                        len(strvalue) > 2 * max_len:
                        right_limit = min(max_len,
                                          len(strvalue)-max_len)
                        value = u'%s ... %s' % (strvalue[:max_len],
                                              strvalue[-right_limit:])
                    string += u"%s%s%s = %s\n" % (
                                        padding, symbol, key, value)
            j += 1
        return string

    def __repr__(self):
        return self._get_print_items().encode('utf8', errors='ignore')

    def __getitem__(self,key):
        return self.__getattribute__(key)
        
    def __setitem__(self,key, value):
        self.__setattr__(key, value)
        
    def __getattribute__(self,name):
        item = super(DictionaryBrowser,self).__getattribute__(name)
        if isinstance(item, dict) and 'value' in item:
            return item['value']
        else:
            return item
            
    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictionaryBrowser(value)
        super(DictionaryBrowser,self).__setattr__(
                         slugify(key, valid_variable_name=True),
                         {'key' : key, 'value' : value})

    def len(self):
        return len(self.__dict__.keys())

    def keys(self):
        return self.__dict__.keys()

    def as_dictionary(self):
        par_dict = {}
        for key_, item_ in self.__dict__.iteritems():
            if type(item_) != types.MethodType:
                key = item_['key']
                if isinstance(item_['value'], DictionaryBrowser):
                    item = item_['value'].as_dictionary()
                else:
                    item = item_['value']
                par_dict.__setitem__(key, item)
        return par_dict
        
    def has_item(self, item_path):
        """Given a path, return True if it exists
        
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by 
            full stops (periods)
            
        Examples
        --------
        
        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False
        
        """
        if type(item_path) is str:
            item_path = item_path.split('.')
        else:
            item_path = copy.copy(item_path)
        attrib = item_path.pop(0)
        if hasattr(self, attrib):
            if len(item_path) == 0:
                return True
            else:
                item = self[attrib]
                if isinstance(item, type(self)): 
                    return item.has_item(item_path)
                else:
                    return False
        else:
            return False
    def __contains__(self, item):
        return self.has_item(item_path=item)
        
    def copy(self):
        return copy.copy(self)
        
    def deepcopy(self):
        return copy.deepcopy(self)
            
    def set_item(self, item_path, value):
        """Given the path and value, create the missing nodes in
        the path and assign to the last one the value
        
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by a 
            full stops (periods)
            
        Examples
        --------
        
        >>> dict_browser = DictionaryBrowser({})
        >>> dict_browser.set_item('First.Second.Third', 3)
        >>> dict_browser
        └── First
           └── Second
                └── Third = 3
        
        """
        if not self.has_item(item_path):
            self.add_node(item_path)
        if type(item_path) is str:
            item_path = item_path.split('.')
        if len(item_path) > 1:
            self.__getattribute__(item_path.pop(0)).set_item(
                item_path, value)
        else:
            self.__setattr__(item_path.pop(), value)



    def add_node(self, node_path):
        """Adds all the nodes in the given path if they don't exist.
        
        Parameters
        ----------
        node_path: str
            The nodes must be separated by full stops (periods).
            
        Examples
        --------
        
        >>> dict_browser = DictionaryBrowser({})
        >>> dict_browser.add_node('First.Second')
        >>> dict_browser.First.Second = 3
        >>> dict_browser
        └── First
            └── Second = 3

        """
        keys = node_path.split('.')
        for key in keys:
            if self.has_item(key) is False:
                self[key] = DictionaryBrowser()
            self = self[key]
            
    
def strlist2enumeration(lst):
    lst = tuple(lst)
    if not lst:
        return ''
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return "%s and %s" % lst
    else:
        return "%s, "*(len(lst) - 2) % lst[:-2] + "%s and %s" % lst[-2:]
        
def ensure_unicode(stuff, encoding = 'utf8', encoding2 = 'latin-1'):
    if type(stuff) is not str and type(stuff) is not np.string_:
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except:
        string = string.decode(encoding2, errors = 'ignore')
    return string
        
def swapelem(obj, i, j):
    """Swaps element having index i with 
    element having index j in object obj IN PLACE.

    E.g.
    >>> L = ['a', 'b', 'c']
    >>> spwapelem(L, 1, 2)
    >>> print L
        ['a', 'c', 'b']
        
    """
    if len(obj) > 1:
        buf = obj[i]
        obj[i] = obj[j]
        obj[j] = buf
        
def rollelem(a, index, to_index=0):
    """Roll the specified axis backwards, until it lies in a given position.
    
    Parameters
    ----------
    a : list
        Input list.
    index : int
        The index of the item to roll backwards.  The positions of the items 
        do not change relative to one another.
    to_index : int, optional
        The item is rolled until it lies before this position.  The default,
        0, results in a "complete" roll.
    
    Returns
    -------
    res : list
        Output list.

    """

    res = copy.copy(a) 
    res.insert(to_index, res.pop(index))
    return res
    
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

def find_subclasses(mod, cls):
    """Find all the subclasses in a module.
    
    Parameters
    ----------
    mod : module
    cls : class
    
    Returns
    -------
    dictonary in which key, item = subclass name, subclass
    
    """
    return dict([(name, obj) for name, obj in inspect.getmembers(mod)
                if inspect.isclass(obj) and issubclass(obj, cls)])
    
def isiterable(obj):
    if isinstance(obj, collections.Iterable):
        return True
    else:
        return False

