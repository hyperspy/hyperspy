import numpy as np
import dask.array as da
from numpy.ma.core import MaskedConstant
from hyperspy.roi import CircleROI
import traits.api as t

masked = MaskedConstant()

def asarray(signal, dtype=None):
    """Convert the input to a masked array of the given data-type.
    Parameters
    -------------
        signal : BaseSignal
    The signal to be converted to a masked signal
    """
    if signal._lazy:
        # changing datatype not allowed
        signal.data = da.ma.asanyarray(signal.data)
    else:
        signal.data = np.ma.asanyarray(signal.data, dtype=dtype)

def masked_equal(signal, value, copy=False):
    """Masks all values equal to some value

    This function is a shortcut to masked_where,
     with condition = (x == value). For floating point arrays,
    consider using masked_values(x, value).
    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        The value to be masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_equal(signal.data, value)
    else:
        signal.data = np.ma.masked_equal(signal.data, value=value, copy=copy)


def masked_greater(signal, value, copy=False):
    """Masks all values greater than some value
    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Data greater than this value will be masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_greater(signal.data, value, copy=copy)
    else:
        signal.data = np.ma.masked_greater(signal.data, value=value, copy=copy)


def masked_greater_equal(signal, value, copy=False):
    """Masks all values greater or equal to some value

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Data greater of equal to this value will be masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_greater_equal(signal.data, value)
    else:
        signal.data = np.ma.masked_greater_equal(signal.data, value=value, copy=copy)

def masked_inside(signal, v1, v2, copy=False):
    """Masks all values inside v1 and v2

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    v1, v2 :
        Values inside v1, and v2 will be masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_inside(signal.data, v1, v2)
    else:
        signal.data = np.ma.masked_inside(signal.data, v1, v2, copy=copy)


def masked_invalid(signal, copy=False):
    """Masks all values set to inf or nan

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    """
    if signal._lazy:
        signal.data = da.ma.masked_invalid(signal.data)
    else:
        signal.data = np.ma.masked_invalid(signal.data,copy=copy)


def masked_less(signal, value, copy=False):
    """Masks all values less than some value

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Data less than value is masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_less(signal.data, value=value)
    else:
        signal.data = np.ma.masked_less(signal.data, value=value, copy=copy)


def masked_less_equal(signal, value, copy=False):
    """Masks all values less than or equal to some value

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Data less than or equal to this value is masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_less_equal(signal.data, value=value)
    else:
        signal.data = np.ma.masked_less_equal(signal.data, value=value, copy=copy)


def masked_not_equal(signal, value, copy=False):
    """Masks all values not equal to some value

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Data not equal to this value is masked
    """
    if signal._lazy:
        signal.data = da.ma.masked_not_equal(signal.data, value=value)
    else:
        signal.data = np.ma.masked_not_equal(signal.data, value=value, copy=copy)


def masked_outside(signal, v1, v2, copy=False):
    """Masks all values outside of v1, and v2

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    v1, v2 :
        Data outside of v1, and v2 are masked.
    """
    if signal._lazy:
        signal.data = da.ma.masked_outside(signal.data, v1, v2)
    else:
        signal.data = np.ma.masked_outside(signal.data, v1, v2, copy=copy)


def masked_values(signal, value, copy=False):
    """Masks all values outside of v1, and v2
    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    value :
        Masking value
    """
    if signal._lazy:
        signal.data = da.ma.masked_values(signal.data, value)
    else:
        signal.data = np.ma.masked_values(signal.data, value, copy=copy)


def masked_where(condition, signal, copy=False):
    """Masks where some condition is met
    Parameters
    -------------
    condition: Boolean Array
        The where the mask should be masked
    signal : BaseSignal
        The signal to be converted to a masked signal
    """
    if signal._lazy:
        signal.data = da.ma.masked_where(condition, signal.data)
    else:
        signal.data = np.ma.masked_where(condition, signal.data, copy=copy)

def masked_roi(signal, roi, axes="signal"):
    """Masks according to some roi

    Parameters
    -------------
    signal : BaseSignal
        The signal to be converted to a masked signal
    roi : hs.roi
        Some ROI defined using hyperspy. 
    axes: "signal", "navigation" or list
        The axes over which to apply the mask
    """
    if axes is None and signal in roi.signal_map:
        axes = roi.signal_map[signal][1]
    else:
        axes = roi._parse_axes(axes, signal.axes_manager)
    natax = signal.axes_manager._get_axes_in_natural_order()

    if isinstance(roi, CircleROI):
        cx = roi.cx + 0.5001 * axes[0].scale
        cy = roi.cy + 0.5001 * axes[1].scale
        ranges = [[cx - roi.r, cx + roi.r],
                  [cy - roi.r, cy + roi.r]]
        x,y = np.ogrid[0:axes[0].size, 0:axes[1].size]
        x = np.subtract(x, cx)
        y = np.subtract(y, cx)
        two_d_mask = (x**2 + y**2) > roi.r ** 2
        print()
        if roi.r_inner != t.Undefined:
            two_d_mask[(x**2 + y**2) < roi.r_inner ** 2] = False
        axes_indexes = (natax.index(axes[0]), natax.index(axes[1]))

        for i in range(len(natax)):
            if i not in axes_indexes:
                two_d_mask = np.expand_dims(two_d_mask,i)
        ones_shape = np.array(signal.axes_manager.shape)
        for i in axes_indexes:
            ones_shape[i]=1
        expanded_ones = np.ones(shape=ones_shape, dtype=bool)
        if signal._lazy:
            expanded_ones = da.array(expanded_ones)
            two_d_mask = da.array(two_d_mask)
            mask = expanded_ones*two_d_mask
        else:
            mask = np.broadcast(expanded_ones,two_d_mask)
        masked_where(mask, signal)
    else:
        ranges = roi._get_ranges()
        slices = roi._make_slices(natax, axes,ranges)
        if signal._lazy: # This is a little memory hungry
            mask = np.zeros(shape=signal.axes_manager.shape,dtype=bool)
            mask[slices] = True
            masked_where(mask,signal)
        else:
            if not isinstance(signal.data, np.ma.masked_array):
                asarray(signal)
            signal.data[slices]=masked






