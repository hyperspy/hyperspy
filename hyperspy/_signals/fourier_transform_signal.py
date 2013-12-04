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

import numpy as np

from hyperspy.signal import Signal

class FourierTransformSignal(Signal):
    _signal_origin = "fourier_transform"
    
    
    def __init__(self, *args, **kw):
        super(FourierTransformSignal,self).__init__(*args, **kw)
    #    self.axes_manager.set_signal_dimension(2)
        
    def ifft(self, s=None, axes=None):
        """
        Compute the N-dimensional inverse discrete Fourier Transform.

        This function computes the inverse of the N-dimensional discrete
        Fourier Transform over any number of axes in an M-dimensional array by
        means of the Fast Fourier Transform (FFT).  In other words,
        ``ifftn(fftn(a)) == a`` to within numerical accuracy.
        For a description of the definitions and conventions used, see `numpy.fft`.

        The input, analogously to `ifft`, should be ordered in the same way as is
        returned by `fftn`, i.e. it should have the term for zero frequency
        in all axes in the low-order corner, the positive frequency terms in the
        first half of all axes, the term for the Nyquist frequency in the middle
        of all axes and the negative frequency terms in the second half of all
        axes, in order of decreasingly negative frequency.

        Parameters
        ----------
        a : array_like
            Input array, can be complex.
        s : sequence of ints, optional
            Shape (length of each transformed axis) of the output
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
            This corresponds to ``n`` for ``ifft(x, n)``.
            Along any axis, if the given shape is smaller than that of the input,
            the input is cropped.  If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input (along the axes specified
            by `axes`) is used.  See notes for issue on `ifft` zero padding.
        axes : sequence of ints, optional
            Axes over which to compute the IFFT.  If not given, the last ``len(s)``
            axes are used, or all axes if `s` is also not specified.
            Repeated indices in `axes` means that the inverse transform over that
            axis is performed multiple times.        

        """
        dim=len(self.axes_manager.shape)
        if dim==1:
            im_ifft=Signal(np.fft.ifft(self.data,s,axes).real)
        elif dim==2:
            im_ifft=Signal(np.fft.ifft2(self.data,s,axes).real)
        else:
            im_ifft=Signal(np.fft.ifftn(self.data,s,axes).real)

        
        if self.axes_manager.signal_dimension==2:
            im_ifft.axes_manager.set_signal_dimension(2)
        #scale?
        for i in range(dim):
            im_ifft.axes_manager[i].scale=1/self.axes_manager[i].scale
            

        #img_ifft=Signal(np.fft.fftn(self.data,s,axes).real)
        return im_ifft
        
    def power_spectrum(self):
        """Compute the power spectrum
        """
        self.data = np.abs(self.data)
    
    def mirror_center(self):
        """Translate the center into the middle
        
        3D
        """
        n=self.data.shape[0]/2
        tmp = self.deepcopy()
        x1=utils.stack([tmp[:n,n+1:,:n],tmp[:n,:n,:n]],axis=1)
        x2=utils.stack([tmp[n+1:,n+1:,:n],tmp[n+1:,:n,:n]],axis=1)
        x3=utils.stack([tmp[:n,n+1:,n+1:],tmp[:n,:n,n+1:]],axis=1)
        x4=utils.stack([tmp[n+1:,n+1:,n+1:],tmp[n+1:,:n,n+1:]],axis=1)
        y1=utils.stack([x3,x1],axis=2)
        y2=utils.stack([x4,x2],axis=2)
        imgn=utils.stack([y2,y1],axis=0)
        return imgn
        
    def rtransform(data,dim,norm=True,n=1): 
        """Radial projection
        
        3D
        """ 
        part1=ones((dim-1,dim-1,dim-1))*power(frange(-(dim-1)/2+0.5,(dim-1)/2-0.5),2)
        dist_mat = power(part1+part1.T+array(map(transpose,part1)),0.5)
        #dist_mat = power(part1+part1.T,0.5)
        #bins = frange(0.,dist_mat[-1,-1,-1],dist_mat[-1,-1,-1]/dim)
        bins = power(frange(0.,power(dist_mat[-1,-1,-1],1.25)/n,
                            power(dist_mat[-1,-1,-1],1.25)/dim*1.5/n),0.8)
        #bins = power(frange(0.,sqrt(dist_mat[-1,-1,-1]),sqrt(dist_mat[-1,-1,-1])/dim),2)
        #bins = power(frange(0.,square(dist_mat[-1,-1,-1])+1,square(dist_mat[-1,-1,-1])/dim),0.5)    
        ydat=[]
        for i in range(len(bins)-1):
            mask_tmp=(dist_mat < bins[i+1]) * (dist_mat > bins[i])
            tmp=data[mask_tmp]
            if norm:
                ydat.append(sum(tmp)/count_nonzero(tmp))
            else:
                ydat.append(sum(tmp))
        return ydat, bins[:-1]

