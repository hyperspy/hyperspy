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

import matplotlib.pyplot as plt
from hyperspy.gui.messages import information


# from pylab import figure, show, cm

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
    
def overwrite(fname):
    """ If file exists 'fname', ask for overwriting and return True or False,
    else return True.
    
    """
    if os.path.isfile(fname):
        message = "Overwrite '%s' (y/n)?\n" % fname
        try:
            answer = raw_input(message)
            answer = answer.lower()
            while (answer != 'y') and (answer != 'n'):
                print('Please answer y or n.')
                answer = raw_input(message)
            if answer.lower() == 'y':
                return True
            elif answer.lower() == 'n':
                # print('Operation canceled.')
                return False
        except:
            # We are running in the IPython notebook that does not
            # support raw_input
            return information(message)
    else:
        return True
    
def saveas(fname):
    """If file 'fname' exists, ask for a new (base)name and, of course, return it.
    
    """
    cdir = os.path.split(fname)[0]
    base = os.path.split(fname)[1]
    if os.path.isfile(fname):
        if overwrite(fname):
            return fname
        else:
            message = 'Please choose a new name.\n'
            answer = raw_input(message)
            newname = os.path.join(cdir, answer)
            return rename(newname)
    else:
        return fname
    
def saveTags(dic, outfile='dm3dataTags.py', val=False):
    """Save keys of dictionary 'dic' in Python format (list)
    into outfile (default: dm3dataTags.py).
    If val == True, the key : value pair is saved.
    """
    print('Saving Tags to', outfile)
    outfile = saveas(outfile)
    if val == False:
        dataList = [ key for key in dic]
        dataList.sort()
    elif val == True:
        dataList = [ (key, dic[key]) for key in dic]
        dataList.sort()
    else:
        print('Invalid option, val=%s' % repr(val))
        return
    with open(outfile, 'w') as fout:
        encoding = 'latin-1'
        fout.write('#!/usr/bin/env python\n')
        fout.write('# -*- coding: %s -*-\n' % encoding)
        fout.write('dataTags = [\n')
        for i in dataList:
            line = "    " + repr(i) + ",\n"
            fout.write(line.encode(encoding))
        fout.write(']\n')
        print 'Done.'

def findInDict(pattern, dic, ig_case=0): # improved from scipy
    """Return a sub-dictionary of dictionary 'dic'
    with all keys containing the regular expression 'pattern'.
    If ig_case = 1, the search is case insensitive.
    """
    result = {}
    if ig_case == 1:
        pat = re.compile(pattern, re.IGNORECASE)
    elif ig_case == 0:
        pat = re.compile(pattern)
    for key in dic:
        if pat.search(key):
            result[key] = dic[key]
    return result

def Unit2Channel(value, origin, scale):
    """Convert calibrated units into uncalibrated units (rounded value):

    uncalibrated: channel = origin + value / scale

    See also: Channel2Unit
    """
    if value is None:
        return None            
    else:
        return int(round(origin + value / scale))

def Channel2Unit(channel, origin, scale):
    """Convert uncalibrated units into calibrated units:

    calibrated: value = (channel - origin) * scale

    See also: Unit2Channel
    """
    if channel is None:
        return None            
    else:
        return (channel - origin) * scale

def getCalibratedPixel(img, *indx):
    """Returns the value of image in a point with calibrated units.

    e.g.
    if img calibration is (x, y, E) => (nm, nm, eV)
    getCalibratedPixel(img, 10.2, 3.2, 50.3)
    will give you the intensity of img @ (10.2 nm, 3.2 nm, 50.3 eV)    
    """
    cpixel = [Unit2Channel(indx[i], img.origin[i], img.scale[i]) for i in xrange(len(indx))]
    
    cpixel = tuple(cpixel)

    return img[cpixel]

class Dimensions(object):
    """Dimensions class characterizes the size, origin, scale and units
    of a given dimension (e.g. image width).
    """
    def __init__(self, size=None, origin=None, scale=None, units=''):
        self.size = size
        self.origin = origin
        self.scale = scale
        self.units = units
        
    def __getattr__(self, attr):
        if attr in ('size', 'origin', 'scale', 'units'):
            return self.__dict__[attr]
        else:
            raise AttributeError, 'attribute "%s" is not allowed.' % attr

    def __setattr__(self, attr, value):
        if attr in ('size', 'origin', 'scale', 'units'):
            self.__dict__[attr] = value
        else:
            raise AttributeError, 'attribute "%s" is not allowed.' % attr

    def __repr__(self):
        return 'Dimensions' + repr((self.size, self.origin, self.scale, self.units))

class IndexTracker(object): # FIXME: check contrast while scrolling
    # modified from original:
    # http://matplotlib.sourceforge.net/examples/pylab_examples/image_slices_viewer.html
    def __init__(self, ax, obj, axis=0, start=None, step=1):
        
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        
        self.obj = obj
        #rows, cols, self.slices = obj.shape
        #self.index = self.slices / 2
        axis = int(axis)
        if not axis in (0, 1, 2):
            raise TypeError, 'Wrong choice of axis (%i)' % axis
        self.axis = axis
        self.slices = obj.shape[self.axis]
        if start is None:
            self.index  = self.slices / 2
        else:
            self.index = start
            
        self.step = step

        if axis == 0:
            sli = self.obj[self.index, :, :]
        elif axis == 1:
            sli = self.obj[:, self.index, :]
        elif axis == 2:
            sli = self.obj[:, :, self.index]
        
        self.im = ax.imshow(sli,
                            interpolation='nearest',
                            cmap=plt.cm.gray)
        self.update()

    def onscroll(self, event):
        if event.button=='up':
            if (self.index + self.step) < self.slices:
                self.index += self.step
            else:
                self.index = self.slices - 1
        elif event.button == 'down':
            if (self.index - self.step) > 0:
                self.index -= self.step
            else:
                self.index = 0
        self.update()

    def update(self):
        if self.axis == 0:
            sli = self.obj[self.index, :, :]
        elif self.axis == 1:
            sli = self.obj[:, self.index, :]
        elif self.axis == 2:
            sli = self.obj[:, :, self.index]        
        self.im.set_data(sli)
        if (self.index + 1) == self.slices:
            self.ax.set_ylabel('LAST SLICE (%s)' % self.index)
        elif self.index == 0:
            self.ax.set_ylabel('FIRST SLICE (%s)' % self.index)
        else:
            self.ax.set_ylabel('Slice n. %s' % self.index)
        
        self.im.axes.figure.canvas.draw()

def slicer(obj, axis=0, start=None, step=1):
    """Displays the slices of a three-dimensional data set
    along a given axis (default is axis 0) as an image and
    allows one to navigate through them using the mouse
    scroll wheel.

    Parameters
    ----------
    obj : three dimensional array.

    axis : int (optional), axis along whom cut the slice.

    start : int (optional), first slice to visualize.
                            Defaults to the middle slice.

    step : int (optional), number of slices to scroll at once.
                           Defaults to 1.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tracker = IndexTracker(ax, obj, axis, start, step)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def ls(dic, pwd, sep='.'):
    """Browse the contents of a dictionary given
    the path in pwd.
    
    pwd can be either a list or a string of keys
    separated by the separator sep (defaults to '.')

    E.g.
    pwd = "root.dir1.dir2.dir3"
    """
    if type(pwd) is str:
        pwd = pwd.split(sep)
    cdir = pwd.pop(0)
    if cdir:
        if pwd:
            return ls(dic[cdir], pwd)
        else:
            try:
                if type(dic[cdir]) is dict:
                    out = dic[cdir].keys()
                    out.sort()
                    return out
                else:
                    return [cdir, ]
            except:
                print("object has no key '%s'" % str(cdir))
    else:
        try:
            out = dic.keys()
            out.sort()
            return out
        except:
            pass

def fsdict(nodes, value, dic):
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
    if node not in dic:
        dic[node] = {}
    if len(nodes) != 0:
        fsdict(nodes,value, dic[node])
    else:
        dic[node] = value
        
def cd(dic, pwd, sep='.'):
    """Return a sub-dictionary
    of dictionary dic given the path in pwd.

    dic should be a dictionary of dictionaries
    
    pwd can be either a list or a string of keys
    separated by the separator sep (defaults to '.')

    E.g.
    pwd = "root.dir1.dir2.dir3"
    """
    if type(pwd) is str:
        pwd = pwd.split(sep)
    cdir = pwd.pop(0)
    if cdir:
        if pwd:
            return cd(dic[cdir], pwd)
        else:
            try:
                if type(dic[cdir]) is dict:
                    return dic[cdir]
                else:
                    print("not a sub-dictionary")
                    return None
            except:
                print("object has no key '%s'" % str(cdir))
    else:
        try:
            return dic
        except:
            pass
     
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
            
