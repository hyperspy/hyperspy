Energy-Dispersive X-Rays
************************

EDS
---

.. versionadded:: 0.7

These methods are only available for the following signals:

* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`


Set elements
^^^^^^^^^^^^

The :py:meth:`~._signals.eds.EDSSpectrum.set_elements` method is used 
to define a set of elements and corresponding X-ray lines
that will be used in other process (e.g. X-ray intensity mapping).
The information is stored in the :py:attr:`~.signal.Signal.mapped_parameters` attribute (see :ref:`mapped_parameters_structure`)


Add elements
^^^^^^^^^^^^

When the set_elements method erases all previously defined elements, 
the :py:meth:`~._signals.eds.EDSSpectrum.add_elements` method adds a new
set of elements to the previous set.


Get intensity map
^^^^^^^^^^^^^^^^^

With the :py:meth:`~._signals.eds.EDSSpectrum.get_intensity_map`, the 
intensity of X-ray lines is used to generate a map. The number of counts
under the selected peaks is used.

Set microscope parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.set_microscope_parameters` method provides an user 
interface to calibrate the paramters if the microscope and the EDS detector.

Get the calibration from another spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.get_calibration_from`
