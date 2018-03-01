import numpy as np
import sparse

from hyperspy.decorators import jit_ifnumba


@jit_ifnumba
def _stream_to_sparse_COO_array_sum_frames(
        stream_data, shape, channels, rebin_energy=1):
    navigation_index = 0
    frame_number = 0
    ysize, xsize = shape
    frame_size = xsize * ysize
    data_list = []
    coords = []
    data = 0
    count_channel = None
    for value in stream_data:
        # when we reach the end of the frame, reset the navigation index to 0
        if navigation_index == frame_size:
            navigation_index = 0
            frame_number += 1
        # if different of ‘65535’, add a count to the corresponding channel
        if value != 65535:  # Same spectrum
            if data:
                if value == count_channel:  # Same channel, add a count
                    data += 1
                else:  # a new channel, same spectrum—requires new coord
                    # Store previous channel
                    coords.append((
                        int(navigation_index // xsize),
                        int(navigation_index % xsize),
                        int(count_channel // rebin_energy))
                    )
                    data_list.append(data)
                    # Add a count to new channel
                    data = 1
                    # Update count channel as this is a new channel
                    count_channel = value

            else:  # First non-zero channel of spectrum
                data = 1
                # Update count channel as this is a new channel
                count_channel = value

        else:  # Advances one pixel
            if data:  # Only store coordinates if the spectrum was not empty
                coords.append((
                    int(navigation_index // xsize),
                    int(navigation_index % xsize),
                    int(count_channel // rebin_energy))
                )
                data_list.append(data)
            navigation_index += 1
            data = 0

    final_shape = (ysize, xsize, channels // rebin_energy)
    coords = np.array(coords)
    data = np.array(data_list)
    return coords.T, data, final_shape


@jit_ifnumba
def _stream_to_sparse_COO_array(stream_data, shape, channels, rebin_energy=1):
    navigation_index = 0
    frame_number = 0
    ysize, xsize = shape
    frame_size = xsize * ysize
    data_list = []
    coords = []
    data = 0
    _coordsf = _get_coords
    count_channel = None
    for value in stream_data:
        # when we reach the end of the frame, reset the navigation index to 0
        if navigation_index == frame_size:
            navigation_index = 0
            frame_number += 1
        # if different of ‘65535’, add a count to the corresponding channel
        if value != 65535:  # Same spectrum
            if data:
                if value == count_channel:  # Same channel, add a count
                    data += 1
                else:  # a new channel, same spectrum—requires new coord
                    # Store previous channel
                    coords.append((
                        frame_number,
                        int(navigation_index // xsize),
                        int(navigation_index % xsize),
                        int(count_channel // rebin_energy))
                    )
                    data_list.append(data)
                    # Add a count to new channel
                    data = 1
                    # Update count channel as this is a new channel
                    count_channel = value

            else:  # First non-zero channel of spectrum
                data = 1
                # Update count channel as this is a new channel
                count_channel = value

        else:  # Advances one pixel
            if data:  # Only store coordinates if the spectrum was not empty
                coords.append((
                    frame_number,
                    int(navigation_index // xsize),
                    int(navigation_index % xsize),
                    int(count_channel // rebin_energy))
                )
                data_list.append(data)
            navigation_index += 1
            data = 0

    final_shape = (frame_number + 1, ysize, xsize, channels // rebin_energy)
    coords = np.array(coords)
    data = np.array(data_list)
    return coords.T, data, final_shape


def stream_to_sparse_COO_array(stream_data, shape, channels, rebin_energy=1,
                               sum_frames=True):
    """Returns data stored in a FEI stream as a nd COO array

    Parameters
    ----------
    stream_data: numpy array
    shape: tuple of ints
        (ysize, xsize)
    channels: ints
        Number of channels in the spectrum
    rebin_energy: int
        Rebin the spectra. The default is 1 (no rebinning applied)
    sum_frames: bool
        If True, sum all the frames

    """

    if sum_frames:
        args = _stream_to_sparse_COO_array_sum_frames(
            stream_data=stream_data,
            shape=shape,
            channels=channels,
            rebin_energy=rebin_energy,)
    else:
        args = _stream_to_sparse_COO_array(
            stream_data=stream_data,
            shape=shape,
            channels=channels,
            rebin_energy=rebin_energy,)
    return sparse.COO(*args)
