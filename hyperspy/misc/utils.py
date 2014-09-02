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

from __future__ import division
import inspect
import copy
import types
import re
from StringIO import StringIO
import codecs
import collections
import tempfile

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
    return np.linspace(
        origin - index * step, origin + step * (N - 1 - index), N)


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
        try:
            # Convert to unicode using the default encoding
            value = unicode(value)
        except:
            # Try latin1. If this does not work an exception is raised.
            value = unicode(value, "latin1")
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(_slugify_strip_re.sub('', value).strip())
    value = _slugify_hyphenate_re.sub('_', value)
    if valid_variable_name is True:
        if value[:1].isdigit():
            value = u'Number_' + value
    return value


class DictionaryTreeBrowser(object):

    """A class to comfortably browse a dictionary using a CLI.

    In addition to accessing the values using dictionary syntax
    the class enables navigating  a dictionary that constains
    nested dictionaries as attribures of nested classes.
    Also it is an iterator over the (key, value) items. The
    `__repr__` method provides pretty tree printing. Private
    keys, i.e. keys that starts with an underscore, are not
    printed, counted when calling len nor iterated.

    Methods
    -------
    export : saves the dictionary in pretty tree printing format in a text file.
    keys : returns a list of non-private keys.
    as_dictionary : returns a dictionary representation of the object.
    set_item : easily set items, creating any necessary node on the way.
    add_node : adds a node.

    Examples
    --------
    >>> tree = DictionaryTreeBrowser()
    >>> tree.set_item("Branch.Leaf1.color", "green")
    >>> tree.set_item("Branch.Leaf2.color", "brown")
    >>> tree.set_item("Branch.Leaf2.caterpillar", True)
    >>> tree.set_item("Branch.Leaf1.caterpillar", False)
    >>> tree
    └── Branch
        ├── Leaf1
        │   ├── caterpillar = False
        │   └── color = green
        └── Leaf2
            ├── caterpillar = True
            └── color = brown
    >>> tree.Branch
    ├── Leaf1
    │   ├── caterpillar = False
    │   └── color = green
    └── Leaf2
        ├── caterpillar = True
        └── color = brown
    >>> for label, leaf in tree.Branch:
            print("%s is %s" % (label, leaf.color))
    Leaf1 is green
    Leaf2 is brown
    >>> tree.Branch.Leaf2.caterpillar
    True
    >>> "Leaf1" in tree.Branch
    True
    >>> "Leaf3" in tree.Branch
    False
    >>>

    """

    def __init__(self, dictionary={}):
        super(DictionaryTreeBrowser, self).__init__()
        self.add_dictionary(dictionary)

    def add_dictionary(self, dictionary):
        """Add new items from dictionary.

        """
        for key, value in dictionary.iteritems():
            self.__setattr__(key, value)

    def export(self, filename, encoding='utf8'):
        """Export the dictionary to a text file

        Parameters
        ----------
        filename : str
            The name of the file without the extension that is
            txt by default
        encoding : valid encoding str

        """
        f = codecs.open(filename, 'w', encoding=encoding)
        f.write(self._get_print_items(max_len=None))
        f.close()

    def _get_print_items(self, padding='', max_len=78):
        """Prints only the attributes that are not methods

        """
        string = ''
        eoi = len(self)
        j = 0
        for key_, value in iter(sorted(self.__dict__.iteritems())):
            if key_.startswith("_"):
                continue
            if not isinstance(key_, types.MethodType):
                key = ensure_unicode(value['key'])
                value = ensure_unicode(value['_dtb_value_'])
                if isinstance(value, DictionaryTreeBrowser):
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
                                          len(strvalue) - max_len)
                        value = u'%s ... %s' % (strvalue[:max_len],
                                                strvalue[-right_limit:])
                    string += u"%s%s%s = %s\n" % (
                        padding, symbol, key, value)
            j += 1
        return string

    def __repr__(self):
        return self._get_print_items().encode('utf8', errors='ignore')

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getattribute__(self, name):
        name = slugify(name, valid_variable_name=True)
        item = super(DictionaryTreeBrowser, self).__getattribute__(name)
        if isinstance(item, dict) and '_dtb_value_' in item and "key" in item:
            return item['_dtb_value_']
        else:
            return item

    def __setattr__(self, key, value):
        slugified_key = str(slugify(key, valid_variable_name=True))
        if isinstance(value, dict):
            if self.has_item(slugified_key):
                self.get_item(slugified_key).add_dictionary(value)
                return
            else:
                value = DictionaryTreeBrowser(value)
        super(DictionaryTreeBrowser, self).__setattr__(
            slugified_key,
            {'key': key, '_dtb_value_': value})

    def __len__(self):
        return len(
            [key for key in self.__dict__.keys() if not key.startswith("_")])

    def keys(self):
        """Returns a list of non-private keys.

        """
        return sorted([key for key in self.__dict__.keys()
                      if not key.startswith("_")])

    def as_dictionary(self):
        """Returns its dictionary representation.

        """
        par_dict = {}
        for key_, item_ in self.__dict__.iteritems():
            if not isinstance(item_, types.MethodType):
                key = item_['key']
                if key == "_db_index":
                    continue
                if isinstance(item_['_dtb_value_'], DictionaryTreeBrowser):
                    item = item_['_dtb_value_'].as_dictionary()
                else:
                    item = item_['_dtb_value_']
                par_dict.__setitem__(key, item)
        return par_dict

    def has_item(self, item_path):
        """Given a path, return True if it exists.

        The nodes of the path are separated using periods.

        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by
            full stops (periods)

        Examples
        --------

        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryTreeBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False

        """
        if isinstance(item_path, str):
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

    def get_item(self, item_path):
        """Given a path, return True if it exists.

        The nodes of the path are separated using periods.

        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by
            full stops (periods)

        Examples
        --------

        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryTreeBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False

        """
        if isinstance(item_path, str):
            item_path = item_path.split('.')
        else:
            item_path = copy.copy(item_path)
        attrib = item_path.pop(0)
        if hasattr(self, attrib):
            if len(item_path) == 0:
                return self[attrib]
            else:
                item = self[attrib]
                if isinstance(item, type(self)):
                    return item.get_item(item_path)
                else:
                    raise AttributeError("Item not in dictionary browser")
        else:
            raise AttributeError("Item not in dictionary browser")

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

        >>> dict_browser = DictionaryTreeBrowser({})
        >>> dict_browser.set_item('First.Second.Third', 3)
        >>> dict_browser
        └── First
           └── Second
                └── Third = 3

        """
        if not self.has_item(item_path):
            self.add_node(item_path)
        if isinstance(item_path, str):
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

        >>> dict_browser = DictionaryTreeBrowser({})
        >>> dict_browser.add_node('First.Second')
        >>> dict_browser.First.Second = 3
        >>> dict_browser
        └── First
            └── Second = 3

        """
        keys = node_path.split('.')
        for key in keys:
            if self.has_item(key) is False:
                self[key] = DictionaryTreeBrowser()
            self = self[key]

    def next(self):
        """
        Standard iterator method, updates the index and returns the
        current coordiantes

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the coordiantes of the current
            iteration.

        """
        if len(self) == 0:
            raise StopIteration
        if not hasattr(self, '_db_index'):
            self._db_index = 0
        elif self._db_index >= len(self) - 1:
            del self._db_index
            raise StopIteration
        else:
            self._db_index += 1
        key = self.keys()[self._db_index]
        return key, getattr(self, key)

    def __iter__(self):
        return self


def strlist2enumeration(lst):
    lst = tuple(lst)
    if not lst:
        return ''
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return "%s and %s" % lst
    else:
        return "%s, " * (len(lst) - 2) % lst[:-2] + "%s and %s" % lst[-2:]


def ensure_unicode(stuff, encoding='utf8', encoding2='latin-1'):
    if not isinstance(stuff, str) and not isinstance(stuff, np.string_):
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except:
        string = string.decode(encoding2, errors='ignore')
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
        fsdict(nodes, value, dictionary[node])
    else:
        dictionary[node] = value


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


def ordinal(value):
    """
    Converts zero or a *postive* integer (or their string
    representations) to an ordinal value.

    >>> for i in range(1,13):
    ...     ordinal(i)
    ...
    u'1st'
    u'2nd'
    u'3rd'
    u'4th'
    u'5th'
    u'6th'
    u'7th'
    u'8th'
    u'9th'
    u'10th'
    u'11th'
    u'12th'

    >>> for i in (100, '111', '112',1011):
    ...     ordinal(i)
    ...
    u'100th'
    u'111th'
    u'112th'
    u'1011th'

    Notes
    -----
    Author:  Serdar Tumgoren
    http://code.activestate.com/recipes/576888-format-a-number-as-an-ordinal/
    MIT license
    """
    try:
        value = int(value)
    except ValueError:
        return value

    if value % 100 // 10 != 1:
        if value % 10 == 1:
            ordval = u"%d%s" % (value, "st")
        elif value % 10 == 2:
            ordval = u"%d%s" % (value, "nd")
        elif value % 10 == 3:
            ordval = u"%d%s" % (value, "rd")
        else:
            ordval = u"%d%s" % (value, "th")
    else:
        ordval = u"%d%s" % (value, "th")

    return ordval


def underline(line, character="-"):
    """Return the line underlined.

    """

    return line + "\n" + character * len(line)


def closest_power_of_two(n):
    return int(2 ** np.ceil(np.log2(n)))


def without_nans(data):
    return data[~np.isnan(data)]


def stack(signal_list, axis=None, new_axis_name='stack_element',
          mmap=False, mmap_dir=None,):
    """Concatenate the signals in the list over a given axis or a new axis.

    The title is set to that of the first signal in the list.

    Parameters
    ----------
    signal_list : list of Signal instances
    axis : {None, int, str}
        If None, the signals are stacked over a new axis. The data must
        have the same dimensions. Otherwise the
        signals are stacked over the axis given by its integer index or
        its name. The data must have the same shape, except in the dimension
        corresponding to `axis`.
    new_axis_name : string
        The name of the new axis when `axis` is None.
        If an axis with this name already
        exists it automatically append '-i', where `i` are integers,
        until it finds a name that is not yet in use.
    mmap: bool
        If True and stack is True, then the data is stored
        in a memory-mapped temporary file.The memory-mapped data is
        stored on disk, and not directly loaded into memory.
        Memory mapping is especially useful for accessing small
        fragments of large files without reading the entire file into
        memory.
    mmap_dir : string
        If mmap_dir is not None, and stack and mmap are True, the memory
        mapped file will be created in the given directory,
        otherwise the default directory is used.

    Returns
    -------
    signal : Signal instance (or subclass, determined by the objects in
        signal list)

    Examples
    --------
    >>> data = np.arange(20)
    >>> s = utils.stack([signals.Spectrum(data[:10]), signals.Spectrum(data[10:])])
    >>> s
    <Spectrum, title: Stack of , dimensions: (2, 10)>
    >>> s.data
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

    """

    axis_input = copy.deepcopy(axis)

    for i, obj in enumerate(signal_list):
        if i == 0:
            if axis is None:
                original_shape = obj.data.shape
                stack_shape = tuple([len(signal_list), ]) + original_shape
                tempf = None
                if mmap is False:
                    data = np.empty(stack_shape,
                                    dtype=obj.data.dtype)
                else:
                    tempf = tempfile.NamedTemporaryFile(
                        dir=mmap_dir)
                    data = np.memmap(tempf,
                                     dtype=obj.data.dtype,
                                     mode='w+',
                                     shape=stack_shape,)

                signal = type(obj)(data=data)
                signal.axes_manager._axes[
                    1:] = copy.deepcopy(
                    obj.axes_manager._axes)
                axis_name = new_axis_name
                axis_names = [axis_.name for axis_ in
                              signal.axes_manager._axes[1:]]
                j = 1
                while axis_name in axis_names:
                    axis_name = new_axis_name + "_%i" % j
                    j += 1
                eaxis = signal.axes_manager._axes[0]
                eaxis.name = axis_name
                eaxis.navigate = True  # This triggers _update_parameters
                signal.metadata = copy.deepcopy(obj.metadata)
                # Get the title from 1st object
                signal.metadata.General.title = (
                    "Stack of " + obj.metadata.General.title)
                signal.original_metadata = DictionaryTreeBrowser({})
            else:
                axis = obj.axes_manager[axis]
                signal = obj.deepcopy()

            signal.original_metadata.add_node('stack_elements')

        # Store parameters
        signal.original_metadata.stack_elements.add_node(
            'element%i' % i)
        node = signal.original_metadata.stack_elements[
            'element%i' % i]
        node.original_metadata = \
            obj.original_metadata.as_dictionary()
        node.metadata = \
            obj.metadata.as_dictionary()

        if axis is None:
            if obj.data.shape != original_shape:
                raise IOError(
                    "Only files with data of the same shape can be stacked")
            signal.data[i, ...] = obj.data
            del obj
    if axis is not None:
        signal.data = np.concatenate([signal_.data for signal_ in signal_list],
                                     axis=axis.index_in_array)
        signal.get_dimensions_from_data()

    if axis_input is None:
        axis_input = signal.axes_manager[-1 + 1j].index_in_axes_manager
        step_sizes = 1
    else:
        step_sizes = [obj.axes_manager[axis_input].size
                      for obj in signal_list]
    signal.metadata._HyperSpy.set_item(
        'Stacking_history.axis',
        axis_input)
    signal.metadata._HyperSpy.set_item(
        'Stacking_history.step_sizes',
        step_sizes)

    return signal
