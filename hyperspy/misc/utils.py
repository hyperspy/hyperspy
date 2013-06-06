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
import copy
import types
import re
from StringIO import StringIO
import codecs

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
        """Prints only the attributes that are not methods"""
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
        
def rollelem(a, index, start = 0):
    """Roll the specified axis backwards, until it lies in a given position.
    
    Parameters
    ----------
    a : list
        Input list.
    index : int
        The index of the item to roll backwards.  The positions of the items 
        do not change relative to one another.
    start : int, optional
        The item is rolled until it lies before this position.  The default,
        0, results in a "complete" roll.
    
    Returns
    -------
    res : list
        Output list.

    """

    res = copy.copy(a) 
    res.insert(start, res.pop(index))
    return res
    

