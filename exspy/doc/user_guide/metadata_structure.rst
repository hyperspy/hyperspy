.. _metadata_structure:

exSpy Metadata Structure
************************

**exSpy** extends the :external+hyperspy:ref:`HyperSpy metadata structure
<metadata_structure>`
with conventions for metadata specific to its signal types. Refer to the
:external+hyperspy:ref:`HyperSpy metadata documentation <metadata_structure>`
for general metadata fields.

The metadata of any **signal objects** is stored in the ``metadata`` attribute,
which has a tree structure. By convention, the node labels are capitalized and
the ones for leaves are not capitalized. When a leaf contains a quantity that
is not dimensionless, the units can be given in an extra leaf with the same
label followed by the ``_units`` suffix.

Besides directly accessing the metadata tree structure, e.g.
``s.metadata.Signal.signal_type``, the HyperSpy methods
:py:meth:`hyperspy.misc.utils.DictionaryTreeBrowser.set_item`,
:py:meth:`hyperspy.misc.utils.DictionaryTreeBrowser.has_item` and
:py:meth:`hyperspy.misc.utils.DictionaryTreeBrowser.get_item`
can be used to add to, search for and read from items in the metadata tree,
respectively.

The EELS of EDS specific metadata structure is represented in the following
tree diagram. The default units are given in parentheses. Details about the
leaves can be found in the following sections of this chapter. Note that not
all types of leaves will apply to every type of measurement.

::

    metadata
    ├── General
    │   └── # see HyperSpy
    ├── Sample
    │   ├── # see HyperSpy
    │   ├── elements
    │   └── xray_lines
    ├── Signal
    │   ├── signal_type
    │   └── # otherwise see HyperSpy
    └── Acquisition_instrument
        ├── SEM
        │   ├── Detector
        │   │   ├── detector_type
        │   │   └── EDS
        │   │       ├── azimuth_angle (º)
        │   │       ├── elevation_angle (º)
        │   │       ├── energy_resolution_MnKa (eV)
        │   │       ├── live_time (s)
        │   │       └── real_time (s)
        │   ├── beam_current (nA)
        │   ├── beam_energy (keV)
        │   ├── probe_area (nm²)
        │   ├── convergence_angle (mrad)
        │   ├── magnification
        │   ├── microscope
        │   ├── Stage
        │   │   ├── rotation (º)
        │   │   ├── tilt_alpha (º)
        │   │   ├── tilt_beta (º)
        │   │   ├── x (mm)
        │   │   ├── y (mm)
        │   │   └── z (mm)
        │   └── working_distance (mm)
        └── TEM
            ├── Detector
            │   ├── EDS
            │   │   ├── azimuth_angle (º)
            │   │   ├── elevation_angle (º)
            │   │   ├── energy_resolution_MnKa (eV)
            │   │   ├── live_time (s)
            │   │   └── real_time (s)
            │   └── EELS
            │       ├── aperture (mm)
            │       ├── collection_angle (mrad)
            │       ├── dwell_time (s)
            │       ├── exposure (s)
            │       ├── frame_number
            │       └── spectrometer
            ├── acquisition_mode
            ├── beam_current (nA)
            ├── beam_energy (keV)
            ├── probe_area (nm²)
            ├── camera_length (mm)
            ├── convergence_angle (mrad)
            ├── magnification
            ├── microscope
            └── Stage
                ├── rotation (º)
                ├── tilt_alpha (º)
                ├── tilt_beta (º)
                ├── x (mm)
                ├── y (mm)
                └── z (mm)


General
=======

See :external+hyperspy:ref:`HyperSpy-Metadata-General <general-metadata>`.

Sample
======

elements
    type: list

    A list of the symbols of the elements composing the sample, e.g. ['B', 'N']
    for a sample composed of Boron and Nitrogen.

xray_lines
    type: list

    A list of the symbols of the X-ray lines to be used for processing,
    e.g. ['Al_Ka', 'Ni_Lb'] for the K alpha line of Aluminum
    and the L beta line of Nickel.

See also :external+hyperspy:ref:`HyperSpy-Metadata-Sample <sample-metadata>`.

Signal
======

signal_type
    type: string

    String that describes the type of signal. Currently, the only exSpy
    specific signal class is ``EELS``, ``EDS``, ``EDS_SEM`` or ``EDS_TEM``.

See also :external+hyperspy:ref:`HyperSpy-Metadata-Sample <signal-metadata>`.

.. _source-metadata:

Acquisition_instrument
======================

TEM
---

Contain information relevant to transmission electron microscope signals.

microscope
    type: Str

    The microscope model, e.g. VG 501

acquisition_mode
    type: Str

    Either 'TEM' or 'STEM'

camera_length
    type: Float

    The camera length in mm.

convergence_angle
    type: Float

    The beam convergence semi-angle in mrad.

beam_energy
    type: Float

    The energy of the electron beam in keV

beam_current
    type: Float

    The beam current in nA.

probe_area
    type: Float

    The illumination area of the electron beam in nm\ :sup:`2`.

dwell_time
    type: Float

    The dwell time in seconds. This is relevant for STEM acquisition

exposure
    type: Float

    The exposure time in seconds. This is relevant for TEM acquisition.

magnification
    type: Float

    The magnification.

SEM
---

Contain information relevant to scanning electron microscope signals.

microscope
    type: Str

    The microscope model, e.g. VG 501

convergence_angle
    type: Float

    The beam convergence semi-angle in mrad.

beam_energy
    type: Float

    The energy of the electron beam in keV

beam_current
    type: Float

    The beam current in nA.

probe_area
    type: Float

    The illumination area of the electron beam in nm\ :sup:`2`.

magnification
    type: Float

    The magnification.

working_distance
    type: Float

    The working distance in mm.


.. _stage-metadata:

Stage
-----
tilt_alpha
    type: Float

    A tilt of the stage in degree.

tilt_beta
    type: Float

    Another tilt of the stage in degree.

rotation
    type: Float

    The rotation of the stage in degree.

x
    type: Float

    The position of the stage in mm along the x axis.

y
    type: Float

    The position of the stage in mm along the y axis.

z
    type: Float

    The position of the stage in mm along the z axis.

.. _detector-metadata:

Detector
--------

All instruments can contain a "Detector" node with information about the
detector used to acquire the signal. EDX and EELS detectors should follow the
following structure:

detector_type
    type: Str

    The type of the detector, e.g. SE for SEM

EELS
^^^^

This node stores parameters relevant to electron energy loss spectroscopy
signals.

aperture_size
    type: Float

    The entrance aperture size of the spectrometer in mm.

collection_angle
    type: Float

    The collection semi-angle in mrad.

dwell_time
    type: Float

    The dwell time in seconds. This is relevant for STEM acquisition

exposure
    type: Float

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
    type: Float

    The azimuth angle of the detector in degree. If the azimuth is zero,
    the detector is perpendicular to the tilt axis.

elevation_angle
    type: Float

    The elevation angle of the detector in degree. The detector is
    perpendicular to the surface with an angle of 90.

energy_resolution_MnKa
    type: Float

    The full width at half maximum (FWHM) of the manganese K alpha
    (Mn Ka) peak in eV. This value is used as a first approximation
    of the energy resolution of the detector.

real_time
    type: Float

    The time spent to record the spectrum in second.

live_time
    type: Float

    The time spent to record the spectrum in second, compensated for the
    dead time of the detector.
