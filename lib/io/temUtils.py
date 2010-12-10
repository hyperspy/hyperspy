#!/usr/bin/env python
# -*- coding: latin-1 -*-

# utility functions for TEM Image suite
#
# Copyright (c) 2010 Stefano Mazzucco.
# All rights reserved.
#
# This program is still at an early stage to be released, so the use of this
# code must be explicitly authorized by its author and cannot be shared for any reason.
#
# Once the program will be mature, it will be released under a GNU GPL license

from __future__ import with_statement #for Python versions < 2.6
import re
from pylab import figure, show, cm

def overwrite(fname):
    """ If file exists, ask for overwriting and return True or False,
    else return True.
    """
    try:
        f = open(fname, 'r')
        message = 'File ' + fname + ' exists. Overwrite (y/n)?\n'
        answer = raw_input(message)
        while (answer != 'y') and (answer != 'n'):
            print 'Please answer y or n.'
            answer = raw_input(message)
        if answer == 'y':
            print 'Writing data to', fname
            f.close()
            return True
        elif answer == 'n':
            print 'Operation canceled.'
            f.close()
            return False
    except IOError: # file does not exist
        return True

def saveTags(dic, outfile='dm3dataTags.py', val=False):
    """Save keys of dictionary 'dic' in Python format (list)
    into outfile (default: dm3dataTags.py).
    If val == True, the key : value pair is saved.
    """
    print 'Saving to', outfile
    exists = overwrite(outfile)
    if exists:
        if val == False:
            dataList = [ key for key in dic]
            dataList.sort()
        elif val == True:
            dataList = [ (key, dic[key]) for key in dic]
            dataList.sort()
        else:
            print 'Invalid option, val=%s' % repr(val)
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
    """
    indx2 = []
    for i in range(len(indx)):
        indx2.append(Unit2Channel(indx[i], img.origin[i], img.scale[i]))
    indx2 = tuple(indx2)

    return img[indx2]

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
        
        self.im = ax.imshow(self.obj[:,:,self.index], interpolation='nearest', cmap=cm.gray)
        self.update()

    def onscroll(self, event):
        # print event.button #, event.step

        if event.button=='up':
            #self.index = np.clip(self.index+1, 0, self.slices-1)
            if (self.index + self.step) < self.slices:
                self.index += self.step
            else:
                self.index = self.slices - 1
        elif event.button == 'down':
            #self.index = np.clip(self.index-1, 0, self.slices-1)
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
    fig = figure()
    ax = fig.add_subplot(111)

    tracker = IndexTracker(ax, obj, start, step)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    show()
