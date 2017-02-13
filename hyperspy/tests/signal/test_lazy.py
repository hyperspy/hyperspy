import pytest
import numpy as np
import dask.array as da
from dask.threaded import get

import hyperspy.api as hs
from hyperspy._signals.lazy import (_reshuffle_mixed_blocks,
                                    to_array)

@pytest.fixture(scope='module')
def signal():
    ar = da.from_array(np.arange(6.*9*7*11).reshape((6,9,7,11)),
                       chunks=((2,1,3), (4,5), (7,), (11,))
                      )
    return hs.signals.LazySignal2D(ar)


@pytest.mark.parametrize("sl", [(0,0), 
                                (slice(None,), 0),
                                (slice(None), slice(None))
                               ]
                        )
def test_reshuffle(signal, sl):
    sig = signal.isig[sl]
    array = np.concatenate(
        [a for a in sig._block_iterator(flat_signal=True,
                                        navigation_mask=None,
                                        signal_mask=None)],
        axis=0
    )
    ndim = sig.axes_manager.navigation_dimension
    ans = _reshuffle_mixed_blocks(array,
                                  ndim,
                                  sig.data.shape[ndim:],
                                  sig.data.chunks[:ndim])
    np.testing.assert_allclose(ans, sig.data.compute())

nav_mask = np.zeros((6,9), dtype=bool)
nav_mask[0,0] = True
nav_mask[1,1] = True
sig_mask = np.zeros((7,11), dtype=bool)
sig_mask[0,:] = True

@pytest.mark.parametrize('nm', [None, nav_mask])
@pytest.mark.parametrize('sm', [None, sig_mask])
@pytest.mark.parametrize('flat', [True, False])
@pytest.mark.parametrize('dtype', ['float', 'int'])
def test_blockiter_bothmasks(signal, flat, dtype, nm, sm):
    real_first = get(signal.data.dask, (signal.data.name, 0, 0, 0 ,0)).copy()
    real_second = get(signal.data.dask, (signal.data.name, 0, 1, 0 ,0)).copy()
    signal.change_dtype(dtype)
    it = signal._block_iterator(flat_signal=flat,
                                navigation_mask=nm,
                                signal_mask=sm,
                                get=get)
    first_block = next(it)
    second_block = next(it)
    if nm is not None:
        nm = nm[:2, :4]
    real_first = real_first.astype(dtype)
    real_second = real_second.astype(dtype)
    if flat:
        if nm is not None:
            nm = ~nm
            navslice = np.where(nm.flat)[0] 
        else:
            navslice = slice(None)
        sigslice = slice(11,None) if sm is not None else slice(None)
        slices1 = (navslice, sigslice)
        real_first = real_first.reshape((2*4, -1))[slices1]
        real_second = real_second.reshape((2*5, -1))[:, sigslice]
    else:
        value = np.nan if dtype is 'float' else 0
        if nm is not None:
            real_first[nm, ...] = value
        if sm is not None:
            real_first[..., sm] = value
            real_second[..., sm] = value
    np.testing.assert_allclose(first_block, real_first)
    np.testing.assert_allclose(second_block, real_second)

@pytest.mark.parametrize('sig', [signal(),
                                 signal().data,
                                 signal().data.compute()])
def test_as_array_numpy(sig):
    thing = to_array(sig, chunks=None)
    assert isinstance(thing, np.ndarray)


@pytest.mark.parametrize('sig', [signal(),
                                 signal().data,
                                 signal().data.compute()])
def test_as_array_dask(sig):
    chunks = ((6,), (9,), (7,), (11,))
    thing = to_array(sig, chunks=chunks)
    assert isinstance(thing, da.Array)
    assert thing.chunks == chunks

def test_as_array_fail():
    with pytest.raises(ValueError):
        to_array('asd', chunks=None)
