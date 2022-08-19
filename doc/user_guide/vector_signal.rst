Vector Signal Tools
*******************

.. versionadded:: 1.8
    BaseVectorSignal class was added to extend the functionality of ragged signals

The methods described in this section are only available for vector signals
signals in the :py:class:`~._signals.vector_signal.BaseVectorSignal`. class.

The Vector class is a new class added in version which is designed to store data which
is organized as a collection of vectors of the same length.  These vectors
represent pixel positions (usually on some signal) and have `VectorDataAxis`
which describes the offset and scale associated with the Axis.

This class is designed to be extendable and closely integrated with methods
in `sklearn` or other packages with advanced tools for vector characterization
and understanding.

Using a Vector Signal
*********************

One of the main uses of a vector signal are finding peaks within some dataset. This method
is great to pair with the find_peaks method or some other workflow which returns a ragged
set of data.

For very large datasets, this is a great way to reduce the size of the dataset and save the data


:: code
