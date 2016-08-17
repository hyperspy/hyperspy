.. _metadata_structure:


Metadata structure
******************

The :class:`~.signal.BaseSignal` class stores metadata in the
:attr:`~.signal.BaseSignal.metadata` attribute that has a tree structure. By
convention, the node labels are capitalized and the leaves are not
capitalized.

When a leaf contains a quantity that has physical dimensions, the units can be
given in an extra leaf with the same label followed by the "_units" suffix. If
no units are specified the default values given in parentheses below are
assumed.

The metadata structure is represented in the following tree diagram. More
detailed descriptions of the contents of the leaves can be found in the
following sections of this chapter.

::

    ├── Acquisition_instrument
    │   ├── SEM
    │   │   ├── Detector
    │   │   │   └── EDS
    │   │   │       ├── azimuth_angle (º)
    │   │   │       ├── elevation_angle (º)
    │   │   │       ├── energy_resolution_MnKa (eV)
    │   │   │       ├── live_time (s)
    │   │   │       └── real_time (s)
    │   │   ├── microscope
    │   │   ├── beam_current (nA)
    │   │   ├── beam_energy (keV)
    │   │   ├── convergence_angle (mrad)
    │       ├── magnification
    │   │   ├── microscope
    │   │   └── tilt_stage (º)
    │   └── TEM
    │       ├── Detector
    │       │   ├── EDS
    │       │   │   ├── azimuth_angle (º)
    │       │   │   ├── elevation_angle (º)
    │       │   │   ├── energy_resolution_MnKa (eV)
    │       │   │   ├── live_time (s)
    │       │   │   └── real_time (s)
    │       │   └── EELS
    │       │       ├── aperture (mm)
    │       │       ├── collection_angle (mrad)
    │       │       ├── dwell_time (s)
    │       │       ├── exposure (s)
    │       │       ├── frame_number
    │       │       └── spectrometer
    │       ├── microscope
    │       ├── acquisition_mode
    │       ├── beam_current (nA)
    │       ├── beam_energy (keV)
    │       ├── camera_length (mm)
    │       ├── convergence_angle (mrad)
    │       ├── magnification
    │       ├── rocking_angle (mrad)
    │       ├── rocking_frequency (Hz)
    │       ├── scan_rotation (º)
    │       └── tilt_stage (º)
    ├── General
    │   ├── authors
    │   ├── date
    │   ├── doi
    │   ├── original_filename
    │   ├── notes
    │   ├── time
    │   └── title
    ├── Sample
    │   ├── description
    │   ├── elements
    │   ├── thickness
    │   └── xray_lines
    └── Signal
        ├── Noise_properties
        │   ├── Variance_linear_model
        │   │   ├── correlation_factor
        │   │   ├── gain_factor
        │   │   ├── gain_offset
        │   │   └── parameters_estimation_method
        │   └── variance
        ├── binned
        ├── quantity
        ├── signal_type
        └── signal_origin

General
=======

title
    type: str

    A title for the signal, e.g. "Sample overview"

original_filename
    type: str

    If the signal was loaded from a file this key stores the name of the
    original file.

time
    type: str

    The acquisition or creation time in ISO 8601 time format.

date
    type: str

    The acquisition or creation date in ISO 8601 date format


authors
    type: Str

    The authors of the data, in Latex format: Surname1, Name1 and Surname2, Name2, etc.

doi
    type: Str

    Digital object identifier of the data, e. g. doi:10.5281/zenodo.58841.

notes
    type: Str

    Notes about the data.

Acquisition_instrument
======================

TEM
---

Contains information relevant to transmission electron microscope signals.

microscope
    type: str

    The microscope model, e.g. VG 501

acquisition_mode
    type: str

    Either 'TEM' or 'STEM'

camera_length
    type: Float

    The camera length in mm.

beam_current
    type: float

    The beam current in nA.

beam_energy
    type: float

    The energy of the electron beam in keV
dwell_time
    type: Float

    The dwell time in seconds. This is relevant for STEM acquisition

exposure
    type: Float

    The exposure time in seconds. This is relevant for TEM acquisition.

magnification
    type: Float

    The magnification.
convergence_angle
    type: float

    The beam convergence semi-angle in mrad.

rocking_angle
    type: float

    The beam rocking semi-angle in mrad.

rocking_frequency
        type: float

        The beam rocking frequency in Hz.

scan_rotation
    type: float

    The rotation of the scan in degrees.

tilt_stage
    type: float

    The tilt of the stage in degrees.


SEM
---

Contains information relevant to scanning electron microscope signals.

microscope
    type: str

    The microscope model, e.g. VG 501

convergence_angle
    type: float

    The beam convergence semi-angle in mrad.

beam_energy
    type: float

    The energy of the electron beam in keV

beam_current
    type: float

    The beam current in nA.

magnification
    type: Float

    The magnification.

tilt_stage
    type: float

    The tilt of the stage in degree.


Detector
--------

All instruments can contain a "Detector" node with information about the
detector used to acquire the signal. EDX and EELS detectors should follow the
following structure:

EELS
^^^^

This node stores parameters relevant to electron energy loss spectroscopy
signals.

spectrometer
    type: str

    The spectrometer model, e.g. Gatan 666

aperture_size
    type: Float

    The entrance aperture size of the spectrometer in mm.
collection_angle
    type: float

    The collection semi-angle in mrad.

dwell_time
    type: float

    The dwell time in seconds. This is relevant for STEM acquisition

exposure
    type: float

    The exposure time in seconds. This is relevant for TEM acquisition.

frame_number
    type: int

    The number of frames/spectra integrated during the acquisition.

spectrometer
    type: Str

    The spectrometer model, e.g. Gatan Enfinium ER (Model 977).


EDS
^^^

This node stores parameters relevant to electron X-ray energy dispersive
spectroscopy data.


azimuth_angle
    type: float

    The azimuth angle of the detector in degree. If the azimuth is zero,
    the detector is perpendicular to the tilt axis.

elevation_angle
    type: float

    The elevation angle of the detector in degree. The detector is perpendicular
    to the surface with an angle of 90.

energy_resolution_MnKa
    type: float

    The full width at half maximum (FWHM) of the manganese K alpha
    (Mn Ka) peak in eV. This value is used as a first approximation
    of the energy resolution of the detector.

real_time
    type: float

    The time spent to record the spectrum in second.

live_time
    type: float

    The time spent to record the spectrum in second, compensated for the
    dead time of the detector.


Diffraction
^^^^^^^^^^^
This node stores parameters relevant to electron diffraction data.


camera_length
    type: float

    The camera length for the acquisition in metres.

exposure_time
    type: float

    The exposure time for the diffraction camera in milliseconds.


Sample
======

description
    type: str

    A brief description of the sample

elements
    type: list

    A list of the symbols of the elements composing the sample, e.g. ['B', 'N']
    for a sample composed of Boron and Nitrogen.

xray_lines
    type: list

    A list of the symbols of the X-ray lines to be used for processing,
    e.g. ['Al_Ka', 'Ni_Lb'] for the K alpha line of Aluminum
    and the L beta line of Nickel.

thickness
    type: float

    The thickness of the sample in m.


Signal
======

signal_type
    type: str

    A term that describes the signal type, e.g. EDS, PES... This information
    can be used by HyperSpy to load the file as a specific signal class and
    therefore the naming should be standarised. Currently HyperSpy provides
    special signal class for photoemission spectroscopy, electron energy
    loss spectroscopy and energy dispersive spectroscopy. The signal_type in these cases should be respectively
    PES, EELS and EDS_TEM (EDS_SEM).

signal_origin
    type: str

    Describes the origin of the signal e.g. 'simulation' or 'experiment'.

record_by
    type: str
    .. deprecated:: 2.1 (HyperSpy v1.0)

    One of 'spectrum' or 'image'. It describes how the data is stored in memory.
    If 'spectrum' the spectral data is stored in the faster index.

quantity
    type: Str
    
    The name of the quantity of the "intensity axis" with the units in round brackets if required, for example Temperature (K).

Noise_properties
----------------

variance
    type: float or BaseSignal instance.

    The variance of the data. It can be a float when the noise is Gaussian or a
    :class:`~.signal.BaseSignal` instance if the noise is heteroscedastic, in which
    case it must have the same dimensions as :attr:`~.signal.BaseSignal.data`.


Variance_linear_model
^^^^^^^^^^^^^^^^^^^^^

In some cases the variance can be calculated from the data using a simple linear
model: ``variance = (gain_factor * data + gain_offset) * correlation_factor``.

gain_factor
    type: float

gain_offset
    type: float

correlation_factor
    type: float

parameters_estimation_method
    type: str


_Internal_parameters
====================

This node is "private" and therefore is not displayed when printing the
:attr:`~.signal.BaseSignal.metadata` attribute. For example, an "energy" leaf
should be accompanied by an "energy_units" leaf.


Stacking_history
----------------

Generated when using :py:meth:`~.utils.stack`. Used by
:py:meth:`~.signal.BaseSignal.split`, to retrieve the former list of signal.

step_sizes
    type: list of int

    Step sizes used that can be used in split.

axis
    type: int

   The axis index in axes manager on which the dataset were stacked.


Folding
-------

Constains parameters that related to the folding/unfolding of signals.
