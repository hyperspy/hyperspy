import tempfile
import os.path
import gc

import numpy as np

import pytest
import numpy.testing as npt

from hyperspy.io import load
import hyperspy.signals as signals
from hyperspy.io_plugins import ripple


# Tuple of tuples (data shape, signal_dimensions)
SHAPES_SDIM = (((3,), (1, )),
               ((2, 3), (1, 2)),
               ((2, 3, 4), (1, 2)),
               )

MYPATH = os.path.dirname(__file__)


def test_write_unsupported_data_shape():
    data = np.arange(5 * 10 * 15 * 20).reshape((5, 10, 15, 20))
    s = signals.Signal1D(data)
    with pytest.raises(IOError):
        s.save('test_write_unsupported_data_shape.rpl')


def test_write_unsupported_data_type():
    data = np.arange(5 * 10 * 15).reshape((5, 10, 15)).astype(np.float16)
    s = signals.Signal1D(data)
    with pytest.raises(IOError):
        s.save('test_write_unsupported_data_type.rpl')


# Test failing
# def test_write_scalar():
#    data = np.array([2])
#    with tempfile.TemporaryDirectory() as tmpdir:
#        s = signals.BaseSignal(data)
#        fname = os.path.join(tmpdir, 'test_write_scalar_data.rpl')
#        s.save(fname)
#        s2 = load(fname)
#        np.testing.assert_allclose(s.data, s2.data)


def test_write_without_metadata():
    data = np.arange(5 * 10 * 15).reshape((5, 10, 15))
    s = signals.Signal1D(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_write_without_metadata.rpl')
        s.save(fname)
        s2 = load(fname)
        np.testing.assert_allclose(s.data, s2.data)
        # for windows
        del s2
        gc.collect()


def test_write_with_metadata():
    data = np.arange(5 * 10).reshape((5, 10))
    s = signals.Signal1D(data)
    s.metadata.set_item('General.date', "2016-08-06")
    s.metadata.set_item('General.time', "10:55:00")
    s.metadata.set_item('General.title', "Test title")
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_write_with_metadata.rpl')
        s.save(fname)
        s2 = load(fname)
        np.testing.assert_allclose(s.data, s2.data)
        assert s.metadata.General.date == s2.metadata.General.date
        assert (s.metadata.General.title == s2.metadata.General.title)
        assert (s.metadata.General.time == s2.metadata.General.time)
        del s2
        gc.collect()


def generate_parameters():
    parameters = []
    for dtype in ripple.dtype2keys.keys():
        for shape, dims in SHAPES_SDIM:
            for dim in dims:
                for metadata in [True, False]:
                    parameters.append({
                        "dtype": dtype,
                        "shape": shape,
                        "dim": dim,
                        "metadata": metadata, })
    return parameters


def _get_filename(s, metadata):
    filename = "test_ripple_sdim-%i_ndim-%i_%s%s.rpl" % (
        s.axes_manager.signal_dimension,
        s.axes_manager.navigation_dimension,
        s.data.dtype.name,
        "_meta" if metadata else "")
    return filename


def _create_signal(shape, dim, dtype, metadata):
    data = np.arange(np.product(shape)).reshape(
        shape).astype(dtype)
    if dim == 1:
        if len(shape) > 2:
            s = signals.EELSSpectrum(data)
            if metadata:
                s.set_microscope_parameters(
                    beam_energy=100.,
                    convergence_angle=1.,
                    collection_angle=10.)
        else:
            s = signals.EDSTEMSpectrum(data)
            if metadata:
                s.set_microscope_parameters(
                    beam_energy=100.,
                    live_time=1.,
                    tilt_stage=2.,
                    azimuth_angle=3.,
                    elevation_angle=4.,
                    energy_resolution_MnKa=5.)
    else:
        s = signals.BaseSignal(data)
        s.axes_manager.set_signal_dimension(dim)
    if metadata:
        s.metadata.General.date = "2016-08-06"
        s.metadata.General.time = "10:55:00"
        s.metadata.General.title = "Test title"
    for i, axis in enumerate(s.axes_manager._axes):
        i += 1
        axis.offset = i * 0.5
        axis.scale = i * 100
        axis.name = "%i" % i
        if axis.navigate:
            axis.units = "m"
        else:
            axis.units = "eV"

    return s


@pytest.mark.parametrize("pdict", generate_parameters())
def test_data(pdict):
    dtype, shape, dim, metadata = (
        pdict["dtype"], pdict["shape"], pdict["dim"], pdict["metadata"])
    with tempfile.TemporaryDirectory() as tmpdir:
        s = _create_signal(
            shape=shape,
            dim=dim,
            dtype=dtype,
            metadata=metadata)
        filename = _get_filename(s, metadata)
        s.save(os.path.join(tmpdir, filename))
        s_just_saved = load(os.path.join(tmpdir, filename))
        s_ref = load(os.path.join(MYPATH, "ripple_files", filename))
        try:
            for stest in (s_just_saved, s_ref):
                npt.assert_array_equal(s.data, stest.data)
                assert s.data.dtype == stest.data.dtype
                assert (s.axes_manager.signal_dimension ==
                        stest.axes_manager.signal_dimension)
                assert (s.metadata.General.title ==
                        stest.metadata.General.title)
                mdpaths = ("Signal.signal_type", )
                if s.metadata.Signal.signal_type == "EELS" and metadata:
                    mdpaths += (
                        "Acquisition_instrument.TEM.convergence_angle",
                        "Acquisition_instrument.TEM.beam_energy",
                        "Acquisition_instrument.TEM.Detector.EELS.collection_angle"
                    )
                elif "EDS" in s.metadata.Signal.signal_type and metadata:
                    mdpaths += (
                        "Acquisition_instrument.TEM.tilt_stage",
                        "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                        "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                        "Acquisition_instrument.TEM.Detector."
                        "EDS.energy_resolution_MnKa",
                        "Acquisition_instrument.TEM.Detector.EDS.live_time",
                    )
                if metadata:
                    mdpaths = (
                        "General.date",
                        "General.time",
                        "General.title",)
                for mdpath in mdpaths:
                    assert (
                        s.metadata.get_item(mdpath) ==
                        stest.metadata.get_item(mdpath))
                for saxis, taxis in zip(
                        s.axes_manager._axes, stest.axes_manager._axes):
                    assert saxis.scale == taxis.scale
                    assert saxis.offset == taxis.offset
                    assert saxis.units == taxis.units
                    assert saxis.name == taxis.name
        except:
            raise
        finally:
            # As of v0.8.5 the data in the ripple files are loaded as memmaps
            # instead of array. In Windows the garbage collector doesn't close
            # the file before attempting to delete it making the test fail.
            # The following lines simply make sure that the memmap is closed.
            # del s_just_saved.data
            # del s_ref.data
            del s_just_saved
            del s_ref
            gc.collect()


def generate_files():
    """Generate the test files that are distributed with HyperSpy.

    Unless new features are introduced there shouldn't be any need to recreate
    the files.

    """
    for dtype in ripple.dtype2keys.keys():
        for shape, dims in SHAPES_SDIM:
            for dim in dims:
                for metadata in [True, False]:
                    s = _create_signal(shape=shape, dim=dim, dtype=dtype,
                                       metadata=metadata)
                    filename = _get_filename(s, metadata)
                    filepath = os.path.join(MYPATH, "ripple_files", filename)
                    s.save(filepath, overwrite=True)
