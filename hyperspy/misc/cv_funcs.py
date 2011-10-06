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
try:
  import cv
except:
  raise ImportError('OpenCV could not be imported')
  
import numpy as np

def cv2array(cv_im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
  
  arrdtype=cv_im.depth
  a = np.fromstring(
         cv_im.tostring(),
         dtype=depth2dtype[cv_im.depth],
         count=cv_im.width*cv_im.height*cv_im.nChannels)
  a.shape = (cv_im.height,cv_im.width,cv_im.nChannels)
  return a
    
def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]), dtype2depth[str(a.dtype)], nChannels)
  cv.SetData(cv_im, a.tostring(),a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im

def xcorr(templateImage,exptImage):
  #cloning is for memory alignment issue with numpy/openCV.
  if type(templateImage).__name__=='ndarray':
    # cast array to 8-bit, otherwise cross correlation fails.
    tmp = templateImage-float(np.min(templateImage))
    tmp = tmp/float(np.max(tmp))
    tmp = np.array(tmp*255,dtype=np.uint8)
    tmp = array2cv(tmp)
  if type(exptImage).__name__=='ndarray':
    expt = exptImage-float(np.min(exptImage))
    expt = expt/float(np.max(expt))
    expt = np.array(expt*255,dtype=np.uint8)
    expt = array2cv(expt)
  tmp=cv.CloneImage(tmp)
  padImage=cv.CloneImage(expt)
  resultWidth = padImage.width - tmp.width + 1
  resultHeight = padImage.height - tmp.height + 1
  result = cv.CreateImage((resultWidth,resultHeight),cv.IPL_DEPTH_32F,1)
  cv.MatchTemplate(padImage,tmp,result,cv.CV_TM_CCOEFF_NORMED)
  result=np.squeeze(cv2array(result))
  return result
