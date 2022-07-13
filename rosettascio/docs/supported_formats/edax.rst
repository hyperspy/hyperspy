.. _edax-format:

EDAX TEAM/Genesis SPD and SPC
-----------------------------

HyperSpy can read both ``.spd`` (spectrum image) and ``.spc`` (single spectra)
files from the EDAX TEAM software and its predecessor EDAX Genesis.
If reading an ``.spd`` file, the calibration of the
spectrum image is loaded from the corresponding ``.ipr`` and ``.spc`` files
stored in the same directory, or from specific files indicated by the user.
If these calibration files are not available, the data from the ``.spd``
file will still be loaded, but with no spatial or energy calibration.
If elemental information has been defined in the spectrum image, those
elements will automatically be added to the signal loaded by HyperSpy.

Currently, loading an EDAX TEAM spectrum or spectrum image will load an
``EDSSEMSpectrum`` Signal. If support for TEM EDS data is needed, please
open an issue in the `issues tracker <https://github.com/hyperspy/hyperspy/issues>`__ to
alert the developers of the need.

For further reference, file specifications for the formats are
available publicly available from EDAX and are on Github
(`.spc <https://github.com/hyperspy/hyperspy/files/29506/SPECTRUM-V70.pdf>`_,
`.spd <https://github.com/hyperspy/hyperspy/files/29505/
SpcMap-spd.file.format.pdf>`_, and
`.ipr <https://github.com/hyperspy/hyperspy/files/29507/ImageIPR.pdf>`_).

Extra loading arguments for SPD file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``spc_fname``: {None, str}, name of file from which to read the spectral calibration. If data was exported fully from EDAX TEAM software, an .spc file with the same name as the .spd should be present. If `None`, the default filename will be searched for. Otherwise, the name of the ``.spc`` file to use for calibration can be explicitly given as a string.
- ``ipr_fname``: {None, str}, name of file from which to read the spatial calibration. If data was exported fully from EDAX TEAM software, an ``.ipr`` file with the same name as the ``.spd`` (plus a "_Img" suffix) should be present.  If `None`, the default filename will be searched for. Otherwise, the name of the ``.ipr`` file to use for spatial calibration can be explicitly given as a string.
- ``**kwargs``: remaining arguments are passed to the Numpy ``memmap`` function.

Extra loading arguments for SPD and SPC files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``load_all_spc`` : bool, switch to control if all of the ``.spc`` header is
  read, or just the important parts for import into HyperSpy.
