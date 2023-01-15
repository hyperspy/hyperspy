.. _storing_models-label:

Storing models
--------------

Multiple models can be stored in the same signal. In particular, when
:py:meth:`~.model.BaseModel.store` is called, a full "frozen" copy of the model
is stored in stored in the signal's :py:class:`~.signal.ModelManager`,
which can be accessed in the ``models`` attribute (i.e. ``s.models``)
The stored models can be recreated at any time by calling
:py:meth:`~.signal.ModelManager.restore` with the stored
model name as an argument. To remove a model from storage, simply call
:py:meth:`~.signal.ModelManager.remove`.

The stored models can be either given a name, or assigned one automatically.
The automatic naming follows alphabetical scheme, with the sequence being (a,
b, ..., z, aa, ab, ..., az, ba, ...).

.. NOTE::

    If you want to slice a model, you have to perform the operation on the
    model itself, not its stored version

.. WARNING::

    Modifying a signal in-place (e.g. :py:meth:`~.signal.BaseSignal.map`,
    :py:meth:`~.signal.BaseSignal.crop`,
    :py:meth:`~._signals.signal1d.Signal1D.align1D`,
    :py:meth:`~._signals.signal2d.Signal2D.align2D` and similar)
    will invalidate all stored models. This is done intentionally.

Current stored models can be listed by calling ``s.models``:

.. code-block:: python

    >>> m = s.create_model()
    >>> m.append(hs.model.components1D.Lorentzian())
    >>> m.store('myname')
    >>> s.models
    └── myname
        ├── components
        │   └── Lorentzian
        ├── date = 2015-09-07 12:01:50
        └── dimensions = (|100)

    >>> m.append(hs.model.components1D.Exponential())
    >>> m.store() # assign model name automatically
    >>> s.models
    ├── a
    │   ├── components
    │   │   ├── Exponential
    │   │   └── Lorentzian
    │   ├── date = 2015-09-07 12:01:57
    │   └── dimensions = (|100)
    └── myname
        ├── components
        │   └── Lorentzian
        ├── date = 2015-09-07 12:01:50
        └── dimensions = (|100)
    >>> m1 = s.models.restore('myname')
    >>> m1.components
       # |      Attribute Name |       Component Name |       Component Type
    ---- | ------------------- | -------------------- | --------------------
       0 |          Lorentzian |           Lorentzian |           Lorentzian


Saving and loading the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To save a model, a convenience function :py:meth:`~.model.BaseModel.save` is
provided, which stores the current model into its signal and saves the
signal. As described in :ref:`storing_models-label`, more than just one
model can be saved with one signal.

.. code-block:: python

    >>> m = s.create_model()
    >>> # analysis and fitting goes here
    >>> m.save('my_filename', 'model_name')
    >>> l = hs.load('my_filename.hspy')
    >>> m = l.models.restore('model_name') # or l.models.model_name.restore()

For older versions of HyperSpy (before 0.9), the instructions were as follows:

    Note that this method is known to be brittle i.e. there is no
    guarantee that a version of HyperSpy different from the one used to save
    the model will be able to load it successfully.  Also, it is
    advisable not to use this method in combination with functions that
    alter the value of the parameters interactively (e.g.
    `enable_adjust_position`) as the modifications made by this functions
    are normally not stored in the IPython notebook or Python script.

    To save a model:

    1. Save the parameter arrays to a file using
       :py:meth:`~.model.BaseModel.save_parameters2file`.

    2. Save all the commands that used to create the model to a file. This
       can be done in the form of an IPython notebook or a Python script.

    3. (Optional) Comment out or delete the fitting commands (e.g.
       :py:meth:`~.model.BaseModel.multifit`).

    To recreate the model:

    1. Execute the IPython notebook or Python script.

    2. Use :py:meth:`~.model.BaseModel.load_parameters_from_file` to load
       back the parameter values and arrays.


Exporting the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~.model.BaseModel` :py:meth:`~.model.BaseModel.export_results`,
:py:class:`~.component.Component` :py:meth:`~.component.Component.export` and
:py:class:`~.component.Parameter` :py:meth:`~.component.Parameter.export`
methods can be used to export the result of the optimization in all supported
formats.
