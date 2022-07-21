.. _protochips-format:

Protochips logfile
------------------

RosettaSciIO can read heater, biasing and gas cell ``.log`` files for Protochips holder.
The format stores all the captured data together with a small header in a ``.csv``
file. The reader extracts the measured quantity (e. g. temperature, pressure,
current, voltage) along the time axis, as well as the notes saved during the
experiment. The reader returns a list of signal with each signal corresponding
to a quantity. Since there is a small fluctuation in the step of the time axis,
the reader assumes that the step is constant and takes its mean, which is a
good approximation. Further releases of RosettaSciIO will read the time axis more
precisely by supporting non-uniform axis (to be implemented!).

To read Protochips logfiles, use the reader argument to define the correct file reader:

.. code-block:: python

    >>> hs.load("filename.csv", reader="protochips")
