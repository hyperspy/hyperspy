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

        
    def ifft(self, s=None, axes=None):
        """
        Compute the inverse discrete Fourier Transform.

        This function computes the inverse of the discrete
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

        s : int or sequence of ints, optional
            Shape (length of each transformed axis) of the output
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
            This corresponds to ``n`` for ``ifft(x, n)``.
            Along any axis, if the given shape is smaller than that of the input,
            the input is cropped.  If it is larger, the input is padded with zeros.
            if `s` is not given, the shape of the input (along the axes specified
            by `axes`) is used.  See notes for issue on `ifft` zero padding.
        axes : int or sequence of ints, optional
            Axes over which to compute the IFFT.  If not given, the last ``len(s)``
            axes are used, or all axes if `s` is also not specified.
            Repeated indices in `axes` means that the inverse transform over that
            axis is performed multiple times.        

        Return
        ------        
        signals.Signal
        
        Notes
        -----        
        For further information see the documentation of numpy.fft.ifft, 
        numpy.fft.ifft2 or numpy.fft.ifftn
        
        """
        dim=len(self.axes_manager.shape)
        if dim==1:
            if axes==None:
                axis=-1
            im_ifft=Signal(np.fft.ifft(self.data,n=s,axis=axis).real)
        elif dim==2:
            if axes==None:
                axes=(-2,-1)
            im_ifft=Signal(np.fft.ifft2(self.data,s=s,axes=axes).real)
        else:
            im_ifft=Signal(np.fft.ifftn(self.data,s=s,axes=axes).real)
        
        if self.axes_manager.signal_dimension==2:
            im_ifft.axes_manager.set_signal_dimension(2)
        #scale,, to be verified
        for i in range(dim):
            im_ifft.axes_manager[i].scale=1/self.axes_manager[i].scale

        return im_ifft

