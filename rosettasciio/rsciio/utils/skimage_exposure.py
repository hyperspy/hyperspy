"""skimage's `rescale_intensity` that takes and returns dask arrays.
"""

import numpy as np
import dask.array as da


from skimage.exposure.exposure import intensity_range, _output_dtype, DTYPE_RANGE




def rescale_intensity(image, in_range='image', out_range='dtype'):
    """Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image. See examples below.

    Parameters
    ----------
    image : array
        Image array.
    in_range, out_range : str or 2-tuple, optional
        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.

        'image'
            Use image min/max as the intensity range.
        'dtype'
            Use min/max of the image's dtype as the intensity range.
        dtype-name
            Use intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`.
        2-tuple
            Use `range_values` as explicit min/max intensities.

    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.

    Notes
    -----
    .. versionchanged:: 0.17
        The dtype of the output array has changed to match the input dtype, or
        float if the output range is specified by a pair of floats.

    See Also
    --------
    equalize_hist

    Examples
    --------
    By default, the min/max intensities of the input image are stretched to
    the limits allowed by the image's dtype, since `in_range` defaults to
    'image' and `out_range` defaults to 'dtype':

    >>> image = np.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)

    It's easy to accidentally convert an image dtype from uint8 to float:

    >>> 1.0 * image
    array([ 51., 102., 153.])

    Use `rescale_intensity` to rescale to the proper range for float dtypes:

    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([0. , 0.5, 1. ])

    To maintain the low contrast of the original, use the `in_range` parameter:

    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([0.2, 0.4, 0.6])

    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:

    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([0.5, 1. , 1. ])

    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter. In that case, the
    output dtype will be float:

    >>> image = np.array([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0. ,  63.5, 127. ])

    To get the desired range with a specific dtype, use ``.astype()``:

    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int8)
    array([  0,  63, 127], dtype=int8)

    If the input image is constant, the output will be clipped directly to the
    output range:
    >>> image = np.array([130, 130, 130], dtype=np.int32)
    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int32)
    array([127, 127, 127], dtype=int32)
    """
    if out_range in ['dtype', 'image']:
        out_dtype = _output_dtype(image.dtype.type, image.dtype)
    else:
        out_dtype = _output_dtype(out_range, image.dtype)

    imin, imax = map(float, intensity_range(image, in_range))
    omin, omax = map(float, intensity_range(image, out_range,
                                            clip_negative=(imin >= 0)))

    if np.any(np.isnan([imin, imax, omin, omax])):
        utils.warn(
            "One or more intensity levels are NaN. Rescaling will broadcast "
            "NaN to the full image. Provide intensity levels yourself to "
            "avoid this. E.g. with np.nanmin(image), np.nanmax(image).",
            stacklevel=2
        )

    image = np.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / (imax - imin)
        return (image * (omax - omin) + omin).astype(dtype=out_dtype)
    else:
        return np.clip(image, omin, omax).astype(out_dtype)