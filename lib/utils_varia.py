#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2007 Francisco Javier de la Peña
# Copyright © 2010 Francisco Javier de la Peña & Stefano Mazzucco
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

# utility functions

from __future__ import with_statement # for Python versions < 2.6

import re

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
    
def overwrite(fname):
    """ If file exists, ask for overwriting and return True or False,
    else return True.
    """
    try:
        f = open(fname, 'r')
        message = 'File ' + fname + ' exists. Overwrite (y/n)?\n'
        answer = raw_input(message)
        while (answer != 'y') and (answer != 'n'):
            print('Please answer y or n.')
            answer = raw_input(message)
        if answer == 'y':
            print('Writing data to', fname)
            f.close()
            return True
        elif answer == 'n':
            print('Operation canceled.')
            f.close()
            return False
    except IOError: # file does not exist
        return True

def saveTags(dic, outfile='dm3dataTags.py', val=False):
    """Save keys of dictionary 'dic' in Python format (list)
    into outfile (default: dm3dataTags.py).
    If val == True, the key : value pair is saved.
    """
    print('Saving Tags to', outfile)
    exists = overwrite(outfile)
    if exists:
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
            print >> fout, '#!/usr/bin/env python'
            print >> fout, '# -*- coding: ' + encoding + ' -*-'
            print >> fout, 'dataTags = ['
            for i in dataList:
                line = "    " + repr(i) + ",\n"
                print >> fout, line.encode(encoding)
            print >> fout, ']'
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
    cpixel = [Unit2Channel(indx[i], img.origin[i], img.scale[i]) for i in range(len(indx))]
    
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
    def __init__(self, ax, obj, start=None, step=1):
        
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        
        self.obj = obj
        #rows, cols, self.slices = obj.shape
        #self.index = self.slices / 2
        self.slices = obj.shape[-1]
        if start is None:
            self.index  = self.slices / 2
        else:
            self.index = start

        self.step = step
        
        self.im = ax.imshow(self.obj[:,:,self.index],
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
        self.im.set_data(self.obj[:, :, self.index])
        if (self.index + 1) == self.slices:
            self.ax.set_ylabel('LAST SLICE (%s)' % self.index)
        elif self.index == 0:
            self.ax.set_ylabel('FIRST SLICE (%s)' % self.index)
        else:
            self.ax.set_ylabel('Slice n. %s' % self.index)
        
        self.im.axes.figure.canvas.draw()

def slicer(obj, start=None, step=1):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tracker = IndexTracker(ax, obj, start, step)

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
                    return dic[cdir].keys()
                else:
                    return [cdir, ]
            except:
                print("object has no key '%s'" % str(cdir))
    else:
        try:
            return dic.keys()
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
                        return self.dic[cdir].keys()
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
                return self.dic.keys()
            except:
                if dbg:
                    msg = 'An error occurred processing '
                    msg += 'the ls() method of '
                    msg += self.__class__.__name__
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
