Vector Signal Tools
*******************

The methods described in this section are only available for vector signals
signals in the :py:class:`~._signals.vector_signal.BaseVectorSignal`. class.

The Vector class is a new class added which is designed to store data which
is organized as a collection of vectors of the same length.  These vectors
represent pixel positions (usually on some signal) and have `VectorDataAxis`
which describes the offset and scale associated with the Axis.

This class is designed to be extendable and closely integrated with methods
in `sklearn` or other packages with advanced tools for vector characterization
and understanding.
