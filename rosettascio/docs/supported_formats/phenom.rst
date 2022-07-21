.. _elid-format:

Phenom ELID format
------------------

The ``.elid`` file format is used by the software package Element Identification for the Thermo
Fisher Scientific Phenom desktop SEM. It is a proprietary binary format which can contain
images, single EDS spectra, 1D line scan EDS spectra and 2D EDS spectrum maps. The reader
will convert all signals and its metadata into dictionaries according to the
RosettaSciIO API.

The current implementation supports ``.elid`` files created with Element Identification version
3.8.0 and later. You can convert older ``.elid`` files by loading the file into a recent Element
Identification release and then save the ``.elid`` file into the newer file format.
