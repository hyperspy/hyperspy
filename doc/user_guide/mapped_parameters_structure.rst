.. _mapped_parameters_structure:


Mapped parameters structure
***************************

General keys
============

These are general parameters that are stored in the root of the 
`mapped_parameters` class.

title
    type: Str
    
    A title for the signal, e.g. "Sample overview"

original_filename
    type: Str
    
    If the signal was loaded from a file this key stores the name of the 
    original file.
    
signal_kind
    type: Str
    
    A term that describes the signal type, e.g. EDX, PES... This information 
    can be used by HyperSpy to load the file as a specific signal class and 
    therefore the naming should be standarised. Currently Hyperspy provides 
    special signal class for photoemission spectroscopy and electron energy 
    loss spectroscopy and the signal_kind in these cases should be respectively 
    PES and EELS.
    
record_by
    type: Str
    
    One of 'spectrum' or 'image'. It describes how the data is stored in memory.
    If 'spectrum' the spectral data is stored in the faster index.
    
time
    type: datetime.time
    
    The acquistion or creation time.
    
date
    type: datetime.time
    
    The acquistion or creation date.
        
Variance_estimation
-------------------

In some cases the variance can be calculated from the data using a simple linear
formula: variance = (gain_factor * I + gain_offset) * correlation_factor
When this is the case, the parameters can be stored in the Variance_estimation
node.

gain_factor
    type: Float

gain_offset
    type: Float

correlation_factor
    type: Float

parameters_estimation_method
    type: Str

Transmission electron microscope keys
-------------------------------------

These keys are stored in the `TEM` node and contain parameters relevant to the 
transmission electron microscope signals. This node is called `SEM`, if the signal
is 'EDS_SEM'.

microscope
    type: Str
    
    The microscope model, e.g. VG 501
    
acquisition_mode
    type: Str
    
    Either 'TEM' or 'STEM'

convergence_angle
    type: Float
    
    The beam convergence angle in mrad.
    
beam_energy
    type: Float
    
    The energy of the electron beam in keV
    
beam_current
    type: Float
    
    The beam current in nA.
    
dwell_time
    type: Float
    
    The dwell time in seconds. This is relevant for STEM acquisition
    
exposure
    type: Float
    
    The exposure time in seconds. This is relevant for TEM acquistion.
    
tilt_stage
    type: Float
    
    The tilt of the stage in degree.
    
Electron energy loss spectroscopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These keys are stored in the `EELS` node of the `TEM` node and contain 
parameters relevant to electron energy loss spectroscopy data.

spectrometer
    type: Str
    
    The spectrometer model, e.g. Gatan 666
    
collection_angle
    type: Float
    
    The collection angle in mrad.
    
X-ray energy dispersive spectroscopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These keys are stored in the `EDS` node of the `TEM` (or `SEM`) node and contain 
parameters relevant to electron X-ray energy dispersive spectroscopy data.


azimuth_angle
    type: Float
    
    The azimuth angle of the detector in degree. If the azimuth is zero,
    the detector is perpendicular to the tilt axis.
    
elevation_angle
    type: Float
    
    The elevation angle of the detector in degree. The detector is perpendicular
    to the surface with an angle of 90.
    
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
   
Sample
------

description
    type: Str
    
    A brief description of the sample
    
elements
    type: list
    
    A list of the symbols of the elements composing the sample, e.g. ['B', 'N'] 
    for a sample composed of Boron and Nitrogen.
    
Xray_lines
    type: list
    
    A list of the symbols of the X-ray lines to be used for processing, 
    e.g. ['Al_Ka', 'Ni_Lb'] for the K alpha line of Aluminum 
    and the L beta line of Nickel.
    
thickness
    type: Float
    
    The thickness of the sample in m.   
    
Stacking_history
----------------

Generated when using :py:meth:`~.utils.stack`. Used by 
:py:meth:`~.signal.Signal.split`, to retrieve the former list of signal.

step_sizes
    type: list of int

    Step sizes used that can be used in split.

axis
    type: int
    
   The axis index in axes manager on which the dataset were stacked.

