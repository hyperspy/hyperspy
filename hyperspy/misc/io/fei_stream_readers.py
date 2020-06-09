# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import numpy as np
import dask.array as da
import sparse

from hyperspy.decorators import jit_ifnumba

# to account for the uncertainty on pixel mapping
pixel_offset = 0

class DenseSliceCOO(sparse.COO):
    """Just like sparse.COO, but returning a dense array on indexing/slicing"""

    def __getitem__(self, *args, **kwargs):
        obj = super().__getitem__(*args, **kwargs)
        try:
            return obj.todense()
        except AttributeError:
            # Indexing, unlike slicing, returns directly the content
            return obj


def stream_to_sparse_COO_array(
        stream_data, spatial_shape, channels, last_frame, rebin_energy=1,
        sum_frames=True, first_frame=0, ):
    """Returns data stored in a FEI stream as a nd COO array

    Parameters
    ----------
    stream_data: numpy array
    spatial_shape: tuple of ints
        (ysize, xsize)
    channels: ints
        Number of channels in the spectrum
    rebin_energy: int
        Rebin the spectra. The default is 1 (no rebinning applied)
    sum_frames: bool
        If True, sum all the frames

    """
    shape = (None, spatial_shape[0], spatial_shape[1], channels)
    if sum_frames:
        coords, data, shape = _stream_to_sparse_summed(stream_data, shape,
                            rebin_energy=rebin_energy,
                            first_frame=first_frame, last_frame=last_frame,)
    else:
        coords, data, shape = _stream_to_sparse(stream_data, shape,
                            rebin_energy=rebin_energy,
                            first_frame=first_frame, last_frame=last_frame,)
    dense_sparse = DenseSliceCOO(coords=coords, data=data, shape=shape)
    dask_sparse = da.from_array(dense_sparse, chunks="auto")
    return dask_sparse


@jit_ifnumba()
def _stream_to_sparse(stream, shape, rebin_energy=1,
                                first_frame=0, last_frame=None):
    """
    Vectorized implementation of stream to sparse conversion. Returns
    coordinates, data, and shape to be turned to a COO matrix

    Parameters
    ----------
    stream: np.array
        Velox emd stream data
    shape: (number of frames, y dimension, x dimension, number of channels)
        number of frames can be none
    rebin_energy: int, optional
        factor to rebin energy axis, must be divisor of number of channels
    first_frame: int, optional
        first frame to consider
    last_frame: int, optional
        last frame to consider
    summed: bool, optional
        Add up data from all frames to one spectrum image

    Returns
    -------
    coords : (frames, ydim, xdim, channel) array
        coordinates where counts are to be added. 
    counts : 1D array
        just an array of ones representing counts
    final_shape: 4 tuple
        (frames, ydim, xdim, channel)
    """
    frms, ydim, xdim, chnx = shape
    # the indexes where counts are registered
    count_channel_regs = np.argwhere(stream != 65535)[:, 0]
    # the pixel index to which these counts must be mapped
    pixel_indexes = (count_channel_regs - np.arange(count_channel_regs.shape[0])
                     - pixel_offset)
    channels = stream[count_channel_regs]//rebin_energy
    # calculate number of frames if it's none
    if frms is None:
        pxl_count = stream.shape[0] - count_channel_regs.shape[0]
        frms = int(np.ceil(pxl_count / xdim / ydim))
    # remove pixel indexes below first frame and above last frame
    first_index = 0  # defaults
    last_index = frms*ydim*xdim
    if first_frame>0:
        first_index = first_frame*ydim*xdim
    if last_frame is not None:
        last_index = last_frame*ydim*xdim
    else:
        last_frame = frms
    filt = (pixel_indexes>=first_index) & (pixel_indexes<last_index)
    pixel_indexes = pixel_indexes[filt]
    channels = channels[filt]
    # if the frames are to be summed the pixel indexes are updated
    final_shape = (last_frame-first_frame, ydim, xdim,
                       chnx//rebin_energy)
    coords = (pixel_indexes//(xdim*ydim)-first_frame,
                  pixel_indexes%(xdim*ydim)//xdim,
                  pixel_indexes%(xdim*ydim)%xdim,
                  channels)
    counts = np.ones(channels.shape[0], dtype=stream.dtype)
    return coords, counts, final_shape


@jit_ifnumba()
def _stream_to_sparse_summed(stream, shape, rebin_energy=1,
                                       first_frame=0, last_frame=None):
    """
    Vectorized implementation of stream to sparse conversion. Returns
    coordinates, data, and shape to be turned to a COO matrix

    Parameters
    ----------
    stream: np.array
        Velox emd stream data
    shape: (number of frames, y dimension, x dimension, number of channels)
        number of frames can be none
    rebin_energy: int, optional
        factor to rebin energy axis, must be divisor of number of channels
    first_frame: int, optional
        first frame to consider
    last_frame: int, optional
        last frame to consider
    summed: bool, optional
        Add up data from all frames to one spectrum image

    Returns
    -------
    coords : (ydim, xdim, channel) array
        coordinates where counts are to be added. shape depends on summed.
    counts : 1D array
        just an array of ones representing counts
    final_shape: 3 tuple
        (ydim, xdim, channel)
    """
    frms, ydim, xdim, chnx = shape
    # the indexes where counts are registered
    count_channel_regs = np.argwhere(stream != 65535)[:, 0]
    # the pixel index to which these counts must be mapped
    pixel_indexes = (count_channel_regs - np.arange(count_channel_regs.shape[0])
                     - pixel_offset)
    channels = stream[count_channel_regs]//rebin_energy
    # calculate number of frames if it's none
    if frms is None:
        pxl_count = stream.shape[0] - count_channel_regs.shape[0]
        frms = int(np.ceil(pxl_count / xdim / ydim))
    # remove pixel indexes below first frame and above last frame
    first_index = 0  # defaults
    last_index = frms*ydim*xdim
    if first_frame>0:
        first_index = first_frame*ydim*xdim
    if last_frame is not None:
        last_index = last_frame*ydim*xdim
    else:
        last_frame = frms
    filt = (pixel_indexes>=first_index) & (pixel_indexes<last_index)
    pixel_indexes = pixel_indexes[filt]
    channels = channels[filt]
    # if the frames are to be summed the pixel indexes are updated
    final_shape = (ydim, xdim, chnx//rebin_energy)
    pixel_indexes = pixel_indexes % (xdim*ydim) 
    coords = (pixel_indexes//xdim,
                pixel_indexes%xdim,
                channels)
    counts = np.ones(channels.shape[0], dtype=stream.dtype)
    return coords, counts, final_shape


def stream_to_array(
        stream, spatial_shape, channels, last_frame, first_frame=0,
        rebin_energy=1, sum_frames=True, dtype="uint16", spectrum_image=None):
    """Returns data stored in a FEI stream as a nd COO array

    Parameters
    ----------
    stream: numpy array
    spatial_shape: tuple of ints
        (ysize, xsize)
    channels: ints
        Number of channels in the spectrum
    rebin_energy: int
        Rebin the spectra. The default is 1 (no rebinning applied)
    sum_frames: bool
        If True, sum all the frames
    dtype: numpy dtype
        dtype of the array where to store the data
    number_of_frame: int or None
    spectrum_image: numpy array or None
        If not None, the array provided will be filled with the data in the
        stream.

    """
    shape = (None, spatial_shape[0], spatial_shape[1], channels)
    if sum_frames:
        coords, data, shape = _stream_to_sparse_summed(stream, shape,
                            rebin_energy=rebin_energy,
                            first_frame=first_frame, last_frame=last_frame,)
    else:
        coords, data, shape = _stream_to_sparse(stream, shape,
                            rebin_energy=rebin_energy,
                            first_frame=first_frame, last_frame=last_frame,)
    return DenseSliceCOO(coords=coords, data=data, shape=shape).todense()


@jit_ifnumba()
def array_to_stream(array):
    """Convert an array to a FEI stream

    Parameters
    ----------
    array: array

    """

    channels = array.shape[-1]
    flat_array = array.ravel()
    stream_data = []
    if pixel_offset == 1:
        stream_data.append(65535)
    channel = 0
    for value in flat_array:
        for j in range(value):
            stream_data.append(channel)
        channel += 1
        if channel % channels == 0:
            channel = 0
            stream_data.append(65535)
    stream_data = stream_data[:-1]  # Remove final mark
    stream_data = np.array(stream_data)
    return stream_data
