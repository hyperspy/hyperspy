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

from operator import attrgetter
import warnings
import inspect
import copy
import types
from io import StringIO
import codecs
import collections
import tempfile
import unicodedata
from contextlib import contextmanager
from ..misc.signal_tools import broadcast_signals
from ..exceptions import VisibleDeprecationWarning

import numpy as np


def attrsetter(target, attrs, value):
    """ Sets attribute of the target to specified value, supports nested
        attributes. Only creates a new attribute if the object supports such
        behaviour (e.g. DictionaryTreeBrowser does)

        Parameters
        ----------
            target : object
            attrs : string
                attributes, separated by periods (e.g.
                'metadata.Signal.Noise_parameters.variance' )
            value : object

        Example
        -------
        First create a signal and model pair:

        >>> s = hs.signals.Signal1D(np.arange(10))
        >>> m = s.create_model()
        >>> m.signal.data
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        Now set the data of the model with attrsetter
        >>> attrsetter(m, 'signal1D.data', np.arange(10)+2)
        >>> self.signal.data
        array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10])

        The behaviour is identical to
        >>> self.signal.data = np.arange(10) + 2


    """
    where = attrs.rfind('.')
    if where != -1:
        target = attrgetter(attrs[:where])(target)
    setattr(target, attrs[where + 1:], value)


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


@contextmanager
def stash_active_state(model):
    active_state = []
    for component in model:
        if component.active_is_multidimensional:
            active_state.append(component._active_array)
        else:
            active_state.append(component.active)
    yield
    for component in model:
        active_s = active_state.pop(0)
        if isinstance(active_s, bool):
            component.active = active_s
        else:
            if not component.active_is_multidimensional:
                component.active_is_multidimensional = True
            component._active_array[:] = active_s


@contextmanager
def dummy_context_manager(*args, **kwargs):
    yield


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


_slugify_strip_re_data = ''.join(
    c for c in map(
        chr, np.delete(
            np.arange(256), [
                95, 32])) if not c.isalnum()).encode()


def slugify(value, valid_variable_name=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    Adapted from Django's "django/template/defaultfilters.py".

    """
    if not isinstance(value, str):
        try:
            # Convert to unicode using the default encoding
            value = str(value)
        except:
            # Try latin1. If this does not work an exception is raised.
            value = str(value, "latin1")
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = value.translate(None, _slugify_strip_re_data).decode().strip()
    value = value.replace(' ', '_')
    if valid_variable_name is True:
        if value[:1].isdigit():
            value = 'Number_' + value
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
    export : saves the dictionary in pretty tree printing format in a text
        file.
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
    ...     print("%s is %s" % (label, leaf.color))
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

    def __init__(self, dictionary=None, double_lines=False):
        self._double_lines = double_lines
        if not dictionary:
            dictionary = {}
        super(DictionaryTreeBrowser, self).__init__()
        self.add_dictionary(dictionary, double_lines=double_lines)

    def add_dictionary(self, dictionary, double_lines=False):
        """Add new items from dictionary.

        """
        for key, value in dictionary.items():
            if key == '_double_lines':
                value = double_lines
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
        from hyperspy.defaults_parser import preferences

        def check_long_string(value, max_len):
            if not isinstance(value, (str, np.string_)):
                value = repr(value)
            value = ensure_unicode(value)
            strvalue = str(value)
            _long = False
            if max_len is not None and len(strvalue) > 2 * max_len:
                right_limit = min(max_len, len(strvalue) - max_len)
                strvalue = '%s ... %s' % (
                    strvalue[:max_len], strvalue[-right_limit:])
                _long = True
            return _long, strvalue

        string = ''
        eoi = len(self)
        j = 0
        if preferences.General.dtb_expand_structures and self._double_lines:
            s_end = '╚══ '
            s_middle = '╠══ '
            pad_middle = '║   '
        else:
            s_end = '└── '
            s_middle = '├── '
            pad_middle = '│   '
        for key_, value in iter(sorted(self.__dict__.items())):
            if key_.startswith("_"):
                continue
            if not isinstance(key_, types.MethodType):
                key = ensure_unicode(value['key'])
                value = value['_dtb_value_']
                if j == eoi - 1:
                    symbol = s_end
                else:
                    symbol = s_middle
                if preferences.General.dtb_expand_structures:
                    if isinstance(value, list) or isinstance(value, tuple):
                        iflong, strvalue = check_long_string(value, max_len)
                        if iflong:
                            key += (" <list>"
                                    if isinstance(value, list)
                                    else " <tuple>")
                            value = DictionaryTreeBrowser(
                                {'[%d]' % i: v for i, v in enumerate(value)},
                                double_lines=True)
                        else:
                            string += "%s%s%s = %s\n" % (
                                padding, symbol, key, strvalue)
                            j += 1
                            continue

                if isinstance(value, DictionaryTreeBrowser):
                    string += '%s%s%s\n' % (padding, symbol, key)
                    if j == eoi - 1:
                        extra_padding = '    '
                    else:
                        extra_padding = pad_middle
                    string += value._get_print_items(
                        padding + extra_padding)
                else:
                    _, strvalue = check_long_string(value, max_len)
                    string += "%s%s%s = %s\n" % (
                        padding, symbol, key, strvalue)
            j += 1
        return string

    def __repr__(self):
        return self._get_print_items()

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getattribute__(self, name):
        if isinstance(name, bytes):
            name = name.decode()
        name = slugify(name, valid_variable_name=True)
        item = super(DictionaryTreeBrowser, self).__getattribute__(name)
        if isinstance(item, dict) and '_dtb_value_' in item and "key" in item:
            return item['_dtb_value_']
        else:
            return item

    def __setattr__(self, key, value):
        if key.startswith('_sig_'):
            key = key[5:]
            from hyperspy.signal import BaseSignal
            value = BaseSignal(**value)
        slugified_key = str(slugify(key, valid_variable_name=True))
        if isinstance(value, dict):
            if self.has_item(slugified_key):
                self.get_item(slugified_key).add_dictionary(
                    value,
                    double_lines=self._double_lines)
                return
            else:
                value = DictionaryTreeBrowser(
                    value,
                    double_lines=self._double_lines)
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
        from hyperspy.signal import BaseSignal
        par_dict = {}
        for key_, item_ in self.__dict__.items():
            if not isinstance(item_, types.MethodType):
                key = item_['key']
                if key in ["_db_index", "_double_lines"]:
                    continue
                if isinstance(item_['_dtb_value_'], DictionaryTreeBrowser):
                    item = item_['_dtb_value_'].as_dictionary()
                elif isinstance(item_['_dtb_value_'], BaseSignal):
                    item = item_['_dtb_value_']._to_dictionary()
                    key = '_sig_' + key
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

    def get_item(self, item_path, default=None):
        """Given a path, return it's value if it exists, or default
        value if missing.

        The nodes of the path are separated using periods.

        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by
            full stops (periods)
        default :
            The value to return if the path does not exist.


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
                    return item.get_item(item_path, default)
                else:
                    return default
        else:
            return default

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
        dtb = self
        for key in keys:
            if dtb.has_item(key) is False:
                dtb[key] = DictionaryTreeBrowser()
            dtb = dtb[key]

    def __next__(self):
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
        key = list(self.keys())[self._db_index]
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
    if not isinstance(stuff, (bytes, np.string_)):
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
    >>> print(L)
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
    '1st'
    '2nd'
    '3rd'
    '4th'
    '5th'
    '6th'
    '7th'
    '8th'
    '9th'
    '10th'
    '11th'
    '12th'

    >>> for i in (100, '111', '112',1011):
    ...     ordinal(i)
    ...
    '100th'
    '111th'
    '112th'
    '1011th'

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
            ordval = "%d%s" % (value, "st")
        elif value % 10 == 2:
            ordval = "%d%s" % (value, "nd")
        elif value % 10 == 3:
            ordval = "%d%s" % (value, "rd")
        else:
            ordval = "%d%s" % (value, "th")
    else:
        ordval = "%d%s" % (value, "th")

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
          lazy=None, **kwargs):
    """Concatenate the signals in the list over a given axis or a new axis.

    The title is set to that of the first signal in the list.

    Parameters
    ----------
    signal_list : list of BaseSignal instances
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
    lazy: {bool, None}
        Returns a LazySignal if True. If None, only returns lazy rezult if at
        least one is lazy.

    Returns
    -------
    signal : BaseSignal instance (or subclass, determined by the objects in
        signal list)

    Examples
    --------
    >>> data = np.arange(20)
    >>> s = hs.stack([hs.signals.Signal1D(data[:10]),
    ...               hs.signals.Signal1D(data[10:])])
    >>> s
    <Signal1D, title: Stack of , dimensions: (2, 10)>
    >>> s.data
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

    """
    from itertools import zip_longest
    from hyperspy.signals import BaseSignal
    import dask.array as da
    from numbers import Number
    # TODO: remove next time
    deprecated = ['mmap', 'mmap_dir']
    warn_str = "'{}' argument is deprecated, please use 'lazy' instead"
    for k in deprecated:
        if k in kwargs:
            lazy=True
            warnings.warn(warn_str.format(k), VisibleDeprecationWarning)

    axis_input = copy.deepcopy(axis)
    signal_list = list(signal_list)
    # Get the real signal with the most axes to get metadata/class/etc
    # first = sorted(filter(lambda _s: isinstance(_s, BaseSignal), signal_list),
    #                key=lambda _s: _s.data.ndim)[-1]
    first = next(filter(lambda _s: isinstance(_s, BaseSignal), signal_list))

    # Cast numbers as signals. Will broadcast later

    for i, _s in enumerate(signal_list):
        if isinstance(_s, BaseSignal):
            pass
        elif isinstance(_s, Number):
            sig = BaseSignal(_s)
            signal_list[i] = sig
        else:
            raise ValueError("{} type cannot be stacked.".format(type(_s)))

    if lazy is None:
        lazy = any(_s._lazy for _s in signal_list)
    if not isinstance(lazy, bool):
        raise ValueError("'lazy' argument has to be None, True or False")

    # Cast all as lazy if required
    for i, _s in enumerate(signal_list):
        if not _s._lazy:
            signal_list[i] = _s.as_lazy()
    if len(signal_list) > 1:
        newlist = broadcast_signals(*signal_list, ignore_axis=axis_input)
        if axis is not None:
            step_sizes = [s.axes_manager[axis].size for s in newlist]
            axis = newlist[0].axes_manager[axis]
        datalist = [s.data for s in newlist]
        newdata = da.stack(datalist, axis=0) if axis is None else \
            da.concatenate(datalist, axis=axis.index_in_array)
        if axis_input is None:
            signal = first.__class__(newdata)
            signal._lazy = True
            signal._assign_subclass()
            signal.axes_manager._axes[1:] = copy.deepcopy(newlist[0].axes_manager._axes)
            axis_name = new_axis_name
            axis_names = [axis_.name for axis_ in signal.axes_manager._axes[1:]]
            j = 1
            while axis_name in axis_names:
                axis_name = new_axis_name + "_%i" % j
                j += 1
            eaxis = signal.axes_manager._axes[0]
            eaxis.name = axis_name
            eaxis.navigate = True  # This triggers _update_parameters
            signal.metadata = copy.deepcopy(first.metadata)
            # Get the title from 1st object
            signal.metadata.General.title = (
                "Stack of " + first.metadata.General.title)
            signal.original_metadata = DictionaryTreeBrowser({})
        else:
            signal = newlist[0]._deepcopy_with_new_data(newdata)
            signal._lazy = True
            signal._assign_subclass()
        signal.get_dimensions_from_data()
        signal.original_metadata.add_node('stack_elements')

        for i, obj in enumerate(signal_list):
            signal.original_metadata.stack_elements.add_node('element%i' % i)
            node = signal.original_metadata.stack_elements['element%i' % i]
            node.original_metadata = \
                obj.original_metadata.as_dictionary()
            node.metadata = \
                obj.metadata.as_dictionary()

        if axis_input is None:
            axis_input = signal.axes_manager[-1 + 1j].index_in_axes_manager
            step_sizes = 1

        signal.metadata._HyperSpy.set_item('Stacking_history.axis', axis_input)
        signal.metadata._HyperSpy.set_item('Stacking_history.step_sizes',
                                           step_sizes)
        if np.all([
                s.metadata.has_item('Signal.Noise_properties.variance')
                for s in signal_list
        ]):
            variance = stack([
                s.metadata.Signal.Noise_properties.variance for s in signal_list
            ], axis)
            signal.metadata.set_item('Signal.Noise_properties.variance', variance)
    else:
        signal = signal_list[0]

    # Leave as lazy or compute
    if lazy:
        signal = signal.as_lazy()
    else:
        signal.compute(False)

    return signal


def shorten_name(name, req_l):
    if len(name) > req_l:
        return name[:req_l - 2] + '..'
    else:
        return name


def transpose(*args, signal_axes=None, navigation_axes=None, optimize=False):
    """Transposes all passed signals according to the specified options.

    For parameters see ``BaseSignal.transpose``.

    Examples
    --------

    >>> signal_iterable = [hs.signals.BaseSignal(np.random.random((2,)*(i+1)))
                           for i in range(3)]
    >>> signal_iterable
    [<BaseSignal, title: , dimensions: (|2)>,
     <BaseSignal, title: , dimensions: (|2, 2)>,
     <BaseSignal, title: , dimensions: (|2, 2, 2)>]
    >>> hs.transpose(*signal_iterable, signal_axes=1)
    [<BaseSignal, title: , dimensions: (|2)>,
     <BaseSignal, title: , dimensions: (2|2)>,
     <BaseSignal, title: , dimensions: (2, 2|2)>]
    >>> hs.transpose(signal1, signal2, signal3, signal_axes=["Energy"])
    """
    from hyperspy.signal import BaseSignal
    if not all(map(isinstance, args, (BaseSignal for _ in args))):
        raise ValueError("Not all pased objects are signals")
    return [sig.transpose(signal_axes=signal_axes,
                          navigation_axes=navigation_axes,
                          optimize=optimize) for sig in args]

def create_map_objects(function, nav_size, iterating_kwargs, **kwargs):
    """To be used in _map_iterate of BaseSignal and LazySignal.

    Moved to a separate method to reduce code duplication.
    """
    from hyperspy.signal import BaseSignal
    from itertools import repeat

    iterators = tuple(signal[1]._iterate_signal()
                      if isinstance(signal[1], BaseSignal) else signal[1]
                      for signal in iterating_kwargs)
    # make all kwargs iterating for simplicity:
    iterating = tuple(key for key, value in iterating_kwargs)
    for k, v in kwargs.items():
        if k not in iterating:
            iterating += k,
            iterators += repeat(v, nav_size),

    def figure_out_kwargs(data):
        _kwargs = {k: v for k, v in zip(iterating, data[1:])}
        for k, v in iterating_kwargs:
            if (isinstance(v, BaseSignal) and
                isinstance(_kwargs[k], np.ndarray) and
                    len(_kwargs[k]) == 1):
                _kwargs[k] = _kwargs[k][0]
        return data[0], _kwargs

    def func(*args):
        dat, these_kwargs = figure_out_kwargs(*args)
        return function(dat, **these_kwargs)

    return func, iterators

def map_result_construction(signal,
                            inplace,
                            result,
                            ragged,
                            sig_shape=None,
                            lazy=False):
    from hyperspy.signals import (BaseSignal, LazySignal)
    res = None
    if inplace:
        sig = signal
    else:
        res = sig = signal._deepcopy_with_new_data()
    if ragged:
        sig.data = result
        sig.axes_manager.remove(sig.axes_manager.signal_axes)
        sig.__class__ = LazySignal if lazy else BaseSignal
        sig.__init__(**sig._to_dictionary(add_models=True))

    else:
        if not sig._lazy and sig.data.shape == result.shape and np.can_cast(
                result.dtype, sig.data.dtype):
            sig.data[:] = result
        else:
            sig.data = result

        # remove if too many axes
        sig.axes_manager.remove(sig.axes_manager.signal_axes[len(sig_shape):])
        # add additional required axes
        for ind in range(
                len(sig_shape) - sig.axes_manager.signal_dimension, 0, -1):
            sig.axes_manager._append_axis(sig_shape[-ind], navigate=False)
    sig.get_dimensions_from_data()
    return res

def multiply(iterable):
    """Return product of sequence of numbers.

    Equivalent of functools.reduce(operator.mul, iterable, 1).

    >>> product([2**8, 2**30])
    274877906944
    >>> product([])
    1

    """
    prod = 1
    for i in iterable:
        prod *= i
    return prod
