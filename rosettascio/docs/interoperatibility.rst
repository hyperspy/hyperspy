================
Interoperability
================

Reading data saved by RosettaSciIO using other software packages
================================================================

The following scripts may help reading data saved using RosettaSciIO using
other software packages, in particular when using the :ref:`HyperSpy HDF5 format
<hspy-format>`.


.. _import-rpl:

ImportRPL Digital Micrograph plugin
-----------------------------------

This Digital Micrograph plugin is designed to import Ripple files into Digital Micrograph.
It is used to ease data transit between DigitalMicrograph and e.g. HyperSpy without losing
the calibration using the extra keywords that HyperSpy adds to the standard format.

When executed, it will ask for 2 files:

#. The ripple file with the data format and calibrations.
#. The data itself in ``.raw`` format.

If a file with the same name and path as the ripple file exists, which has a
``.raw`` or ``.bin`` extension, it is opened directly without prompting.
ImportRPL was written by Luiz Fernando Zagonel.

Download ``ImportRPL`` from the `ImportRPL GitHub repository 
<https://github.com/hyperspy/ImportRPL>`_


.. _dm-import-hdf5:

HDF5 reader plugin for Digital Micrograph
-----------------------------------------

This Digital Micrograph plugin is designed to import HDF5 files and like the
`ImportRPL` script above, it can be used to easily transfer data saved using
RosettaSciIO to Digital Micrograph by using the HDF5 files following the
:ref:`HyperSpy format specification <hspy-format>` (``.hspy`` extension).

Download ``gms_plugin_hdf5`` from the `gms_plugin_hdf5 GitHub repository
<https://github.com/niermann/gms_plugin_hdf5>`_.


.. _hyperspy-matlab:

readHyperSpyH5 MATLAB Plugin
----------------------------

This MATLAB script is designed to import HDF5 files saved using the :ref:`HyperSpy
format specification <hspy-format>` (``.hspy`` extension).
Like the Digital Micrograph script above, it is used to easily transfer data
saved using RosettaSciIO to MATLAB, while e.g. retaining spatial calibration
information from HyperSpy.

Download ``readHyperSpyH5`` from the `readHyperSpyH5 GitHub repository <https://github.com/jat255/readHyperSpyH5>`_.
