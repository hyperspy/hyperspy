.. _dens-format:

DENSsolutions formats
---------------------
HyperSpy can read any logfile from DENSsolutions' new Impulse software as well
as the legacy heating software DigiHeater.

.. _dens_impulse-format:

DENSsolutions Impulse logfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impulse logfiles are stored in ``.csv`` format. All metadata linked to the experiment
is stored in a separate ``metadata.log`` file. This metadata file contains crucial
information about the experiment and should be included in the same folder with
the ``.csv`` file when reading data into Hyperspy.

To read Impulse logfiles, use the reader argument to define the correct file reader:

.. code-block:: python

    >>> hs.load("filename.csv", reader="impulse")


.. _dens_heater-format:

DENSsolutions DigiHeater logfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy can read the heater log format from the DENSsolutions’ DigiHeater software.
The format stores all the captured data for each timestamp, together with a small
header in a plain-text format. The reader extracts the measured temperature along
the time axis, as well as the date and calibration constants stored in the header.
