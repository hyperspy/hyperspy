#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Copyright (C) 2011 by Michael Sarahan

 This file is part of EELSLab.

 EELSLab is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 EELSLab is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with EELSLab; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
 USA

"""

import sys
try:
    import cv
except:
    # commented out because just need one_dim_findpeks
    # print "OpenCV not available!  Exiting."
    # sys.exit()
    print 'Module %s:' % sys.modules[__name__]
    print 'OpenCV is not available, most of the functions will not work.'
import numpy as np
from scipy.signal import medfilt

def one_dim_findpeaks(y, x=None, slope_thresh=0.5, amp_thresh=None,
              medfilt_radius=5, maxpeakn=30000, peakgroup=10, subpix=True):
    """
    Find peaks along a 1D line.

    Function to locate the positive peaks in a noisy x-y data set.
    
    Detects peaks by looking for downward zero-crossings in the first
    derivative that exceed 'slope_thresh'.
    
    Returns an array containing position, height, and width of each peak.

    'slope_thresh' and 'amp_thresh', control sensitivity: higher values will
    neglect smaller features.
    
    peakgroup is the

    Parameters
    ---------
    y : array
        1D input array, e.g. a spectrum
        
    x : array (optional)
        1D array describing the calibration of y (must have same shape as y)

    slope_thresh : float (optional)
                   1st derivative threshold to count the peak
                   default is set to 0.5
                   higher values will neglect smaller features.
                   
    amp_thresh : float (optional)
                 intensity threshold above which   
                 default is set to 10% of max(y)
                 higher values will neglect smaller features.
                                  
    medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5

    peakgroup : int (optional)
                number of points around the "top part" of the peak
                default is set to 10

    maxpeakn : int (optional)
              number of maximum detectable peaks
              default is set to 30000
                
    subpix : bool (optional)
             default is set to True

    Returns
    -------
    P : array of shape (npeaks, 3)
        contains position, height, and width of each peak
    
    """
    # Changelog
    # T. C. O'Haver, 1995.  Version 2  Last revised Oct 27, 2006
    # Converted to Python by Michael Sarahan, Feb 2011.
    # Revised to handle edges better.  MCS, Mar 2011
    if x is None:
        x = np.arange(len(y),dtype=np.int64)
    if not amp_thresh:
        amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius:
        d = np.gradient(medfilt(y,medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    # allocate a result array for 'maxpeakn' peaks
    P = np.zeros((maxpeakn, 3))
    peak = 0
    for j in xrange(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j+1]): # Detects zero-crossing
            if np.sign(d[j+1]) == 0: continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j+1] > slope_thresh:
                # if height of peak is larger than amp_thresh
                if y[j] > amp_thresh:  
                    # the next section is very slow, and actually messes
                    # things up for images (discrete pixels),
                    # so by default, don't do subpixel precision in the
                    # 1D peakfind step.
                    if subpix:
			xx = np.zeros(peakgroup)
			yy = np.zeros(peakgroup)
			s = 0
			for k in xrange(peakgroup): 
			    groupindex = j + k - n + 1 
			    if groupindex < 1:
				xx = xx[1:]
				yy = yy[1:]
				s += 1
				continue
			    elif groupindex > y.shape[0] - 1:
				xx = xx[:groupindex-1]
				yy = yy[:groupindex-1]
				break
			    xx[k-s] = x[groupindex]
			    yy[k-s] = y[groupindex]
			avg = np.average(xx)
			stdev = np.std(xx)
			xxf = (xx - avg) / stdev
			# Fit parabola to log10 of sub-group with
                        # centering and scaling
			coef = np.polyfit(xxf, np.log10(np.abs(yy)), 2)  
			c1 = coef[2]
			c2 = coef[1]
			c3 = coef[0]
			width = np.linalg.norm(
                            stdev * 2.35703 / (np.sqrt(2) * np.sqrt(-1 * c3)))
			# if the peak is too narrow for least-squares
                        # technique to work  well, just use the max value
                        # of y in the sub-group of points near peak.
			if peakgroup < 7:
			    height = np.max(yy)
			    location = xx[np.argmin(np.abs(yy - height))]
			else:
                            location =- ((stdev * c2 / (2 * c3)) - avg)
			    height = np.exp( c1 - c3 * (c2 / (2 * c3))**2)    
                    # Fill results array P. One row for each peak 
                    # detected, containing the
                    # peak position (x-value) and peak height (y-value).
		    else:
			location = x[j]
			height = y[j]
                        # no way to know peak width without
                        # the above measurements.
			width = 0
                    if (location > 0 and not np.isnan(location)
                        and location < x[-1]):
                        P[peak] = np.array([location, height, width])
                        peak = peak + 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    return P[:peak,:]

def two_dim_findpeaks(arr,subpixel=False,peak_width=10,medfilt_radius=5):
    """
    Locate peaks on a 2-D image.  Basic idea is to locate peaks in X direction,
    then in Y direction, and see where they overlay.

    Code based on Dan Masiel's matlab functions
    """
    #
    mapX=np.zeros_like(arr)
    mapY=np.zeros_like(arr)
    arr=medfilt(arr,medfilt_radius)
    xc = [one_dim_findpeaks(arr[i], medfilt_radius=None,
                             peakgroup=boxsize,
                             subpix=False)[:,0] for i in xrange(arr.shape[1])]
    for row in xrange(len(xc)):
        for col in xrange(xc[row].shape[0]):
            mapX[row,int(xc[row][col])]=1
    yc = [one_dim_findpeaks(arr[:,i], medfilt_radius=None,
                             peakgroup=boxsize,
                             subpix=False)[:,0] for i in xrange(arr.shape[0])]
    for row in xrange(len(yc)):
        for col in xrange(yc[row].shape[0]):
            mapY[row,int(yc[row][col])]=1
    # Dan's comment from Matlab code, left in for curiosity:
    #% wow! lame!
    Fmap = mapX*mapY.T
    nonzeros=np.nonzero(Fmap)
    coords=np.vstack((nonzeros[1],nonzeros[0])).T
    if subpixel:
        coords=subpix_locate(arr,coords,boxsize)
    coords=np.ma.fix_invalid(coords,fill_value=-1)
    coords=np.ma.masked_outside(coords,boxsize/2+1,arr.shape[0]-boxsize/2-1)
    coords=np.ma.masked_less(coords,0)
    coords=np.ma.compress_rows(coords)
    return coords 

def subpix_locate(data,points,boxsize,scale=None):
    from scipy.ndimage.measurements import center_of_mass as CofM
    top=left=boxsize/2
    centers=np.array(points,dtype=np.float32)
    for i in xrange(points.shape[0]):
        pt=points[i]
        center=np.array(CofM(data[(pt[0]-left):(pt[0]+left),(pt[1]-top):(pt[1]+top)]))
        center=center[0]-boxsize/2,center[1]-boxsize/2
        centers[i]=np.array([pt[0]+center[0],pt[1]+center[1]])
    if scale:
        centers=centers*scale
    return centers
    
def stack_coords(stack,peakwidth,subpixel=False):
    """
    A rough location of all peaks in the image stack.  This can be fed into the
    best_match function with a list of specific peak locations to find the best
    matching peak location in each image.
    """
    depth=stack.shape[2]
    coords=np.ones((5000,2,depth))*10000
    for i in xrange(depth):
        ctmp=two_dim_findpeaks(stack[:,:,i], subpixel=subpixel,
                               boxsize=peakwidth)
        for row in xrange(ctmp.shape[0]):
            coords[row,:,i]=ctmp[row]
    return coords
    
def best_match(arr,target,neighborhood=None):
    """
    Attempts to find the best match for target in array arr.  Assumes a 3D array,
    consisting of peak coordinates from each image.  Each row is a peak coordinate, 
    the first column is x, the second column is y, and each layer of depth
    is a separate image.

    Returns an array with the best matching coordinates for each image.
    
    Usage:
        best_match(arr, target)
    """
    depth=arr.shape[2]
    rlt=np.zeros((depth,2))
    arr_sub=arr.copy()
    for d in xrange(depth):
        arr_sub[:,:,d]=arr[:,:,d]-target
    if neighborhood:
        arr_sub=np.ma.masked_outside(arr_sub,-neighborhood,neighborhood)
        arr_sub=np.ma.filled(arr_sub,10000)
    # Calculate euclidean distance for each peak in arr_sub; locate minimum
    matches=[np.argmin(
                np.sqrt(np.sum(
                    np.power(arr_sub[:,:,i],2),
                    axis=1))
                ) for i in xrange(depth)]
    # Fill result array with found minima
    for i in xrange(depth):
        rlt[i]=arr[matches[i],:,i]
        if neighborhood:
            if np.sqrt(np.sum(np.power(rlt[i]-target,2)))>neighborhood:
                print """
Warning! Peak for image %i at target (%i,%i) was outside specified neighborhood!
"""%(i,target[0],target[1])
    return rlt
  
def peak_attribs(image,locations,peakwidth,medfilt_radius=5):
    rlt=np.zeros((locations.shape[0],5))
    r=peakwidth/2
    imsize=image.shape[0]
    roi=np.zeros((peakwidth,peakwidth))
    if medfilt_radius:
        image=medfilt(image,medfilt_radius)
    for loc in xrange(locations.shape[0]):
        c=locations[loc]
        bxmin=c[0]-r
        bymin=c[1]-r
        bxmax=c[0]+r
        bymax=c[1]+r
        if bxmin<0: bxmin=0; bxmax=peakwidth
        if bymin<0: bymin=0; bymax=peakwidth
        if bxmax>imsize: bxmax=imsize; bxmin=imsize-peakwidth
        if bymax>imsize: bymax=imsize; bymin=imsize-peakwidth
        roi[:,:]=image[bxmin:bxmax,bymin:bymax]
        ms=cv.Moments(cv.fromarray(roi))
        height=np.max(roi)
        orient=orientation(ms)
        ecc=eccentricity(ms)
        rlt[loc,:2]=c
        rlt[loc,2]=height
        rlt[loc,3]=orient
        rlt[loc,4]=ecc
    return rlt

def peak_attribs_stack(stack, locations, peakwidth, medfilt_radius=5, imcoords=None):
    """
    Given a stack of images and a list of locations and window sizes (defined by
    the peak width), measure the peak attributes of the peaks of interest in
    each image.  These attributes currently include the height, location of the 
    peak and the relative difference in position of the peak from the average, 
    peak orientation angle and eccentricity.
    """
    # pre-allocate result array.  7 rows for each peak, 1 column for each image
    if imcoords:
        # an extra row for keeping track of image coordinates
        rlt=np.zeros((7*avg_locations.shape[0],stack.shape[1]+1))
    else:
        rlt=np.zeros((7*avg_locations.shape[0],stack.shape[1]))
    rlt_tmp=np.zeros(locations.shape[0],5)
    for i in xrange(stack.shape[2]):
        rlt_tmp=peak_attribs(stack[:,:,i], locations, peakwidth, medfilt_radius)
        diff_coords=locations-rlt_tmp[:,:2]
        for j in xrange(locations.shape[0]):
            rlt[j*7:j*7+2,i]=rlt_tmp[j,:2]
            rlt[j*7+2:j*7+4,i]=diff_coords[j]
            rlt[j*7+4]=rlt_tmp[j,2]
            rlt[j*7+5]=rlt_tmp[j,3]
            rlt[j*7+6]=rlt_tmp[j,4]
    if imcoords is not None:
        if imcoords.shape[0]==rlt.shape[1]/7:
            pass
        else:
            imcoords=imcoords.T
        for im in xrange(imcoords.shape[0]):
            rlt[0,im*7:im*7+2]=imcoords[im]
    return rlt
    
def normalize(arr,lower=0.0,upper=1.0):
    if lower>upper: lower,upper=upper,lower
    arr -= arr.min()
    arr *= (upper-lower)/arr.max()
    arr += lower
    return arr

def center_of_mass(moments):
    x = cv.GetCentralMoment(moments,1,0)/cv.GetCentralMoment(moments,0,0)
    y = cv.GetCentralMoment(moments,0,1)/cv.GetCentralMoment(moments,0,0)
    return x,y
            
def orientation(moments):
    mu11p = cv.GetCentralMoment(moments,1,1)/cv.GetCentralMoment(moments,0,0)
    mu02p = cv.GetCentralMoment(moments,2,0)/cv.GetCentralMoment(moments,0,0)
    mu20p = cv.GetCentralMoment(moments,0,2)/cv.GetCentralMoment(moments,0,0)
    return 0.5*np.arctan(2*mu11p/(mu20p-mu02p))

def eccentricity(moments):
    mu11p = cv.GetCentralMoment(moments,1,1)/cv.GetCentralMoment(moments,0,0)
    mu02p = cv.GetCentralMoment(moments,2,0)/cv.GetCentralMoment(moments,0,0)
    mu20p = cv.GetCentralMoment(moments,0,2)/cv.GetCentralMoment(moments,0,0)
    return ((mu20p-mu02p)**2-4*mu11p**2)/(mu20p+mu02p)**2

def characterize_stack(stack, peak_width, neighborhood=None, locs=None, 
                       subpixel=False, medfilt_radius=5):
    if locs is None:
        avg_image=np.average(stack,axis=2)
        locs=two_dim_findpeaks(avg_image, subpixel, peak_width, medfilt_radius)
    cstack=stack_coords(stack,peak_width,subpixel)
    best_stack=np.array([best_match(cstack,loc,neighborhood) for loc in locs])
    attribs=peak_attribs_stack(stack,best_stack,peak_width)
    return attribs

if __name__=='__main__':
    from io import image_stack
    from glob import glob
    flist=glob('*.png')
    peak_width=8
    neighborhood=20
    d=image_stack.read(flist)
    attribs=characterize_stack(d, peak_width, neighborhood, subpixel=True, medfilt_radius=5)
    np.save('attribs.npy',attribs)
    
