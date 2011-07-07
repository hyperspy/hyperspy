# -*- coding: utf-8 -*-
"""
Copyright (C) 2011 by Michael Sarahan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cv
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
