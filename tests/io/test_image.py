# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest

import hyperspy.api as hs


try:
    from matplotlib_scalebar.scalebar import ScaleBar
    matplotlib_scalebar_installed = True
except ImportError:  # pragma: no cover
    matplotlib_scalebar_installed = False


@pytest.mark.parametrize(("dtype"), ['uint8', 'uint32'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpg'])
def test_save_load_cycle_grayscale(dtype, ext, tmp_path):
    s = hs.signals.Signal2D(np.arange(128*128).reshape(128, 128).astype(dtype))

    print('Saving-loading cycle for the extension:', ext)
    filename = tmp_path / f'test_image.{ext}'
    s.save(filename)
    hs.load(filename)


@pytest.mark.parametrize(("color"), ['rgb8', 'rgba8', 'rgb16', 'rgba16'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpeg'])
def test_save_load_cycle_color(color, ext, tmp_path):
    dim = 4 if "rgba" in color else 3
    dtype = 'uint8' if "8" in color else 'uint16'
    if dim == 4 and ext == 'jpeg':
        # JPEG does not support alpha channel.
        return
    print('color:', color, '; dim:', dim, '; dtype:', dtype)
    s = hs.signals.Signal1D(np.arange(128*128*dim).reshape(128, 128, dim).astype(dtype))
    s.change_dtype(color)

    print('Saving-loading cycle for the extension:', ext)
    filename = tmp_path / f'test_image.{ext}'
    s.save(filename)
    hs.load(filename)


@pytest.mark.parametrize(("dtype"), ['uint8', 'uint32'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpg'])
def test_save_load_cycle_kwds(dtype, ext, tmp_path):
    s = hs.signals.Signal2D(np.arange(128*128).reshape(128, 128).astype(dtype))

    print('Saving-loading cycle for the extension:', ext)
    filename = tmp_path / f'test_image.{ext}'
    if ext == 'png':
        if dtype == 'uint32':
            kwds = {'bits': 32}
        else:
            kwds = {'optimize': True}
    elif ext == 'jpg':
        kwds = {'quality': 100, 'optimize': True}
    elif ext == 'gif':
        kwds = {'subrectangles': 'True', 'palettesize': 128}
    else:
        kwds = {}
    s.save(filename, **kwds)
    hs.load(filename, pilmode='L', as_grey=True)


@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpg'])
def test_export_scalebar(ext, tmp_path):
    data = np.arange(1E6).reshape((1000, 1000))
    s = hs.signals.Signal2D(data)
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'nm'

    filename = tmp_path / f'test_scalebar_export.{ext}'
    if ext in ['bmp', 'gif'] and matplotlib_scalebar_installed:
        with pytest.raises(ValueError):
            s.save(filename, scalebar=True)
        with pytest.raises(ValueError):
            s.save(filename, output_size=512)
        s.save(filename)
    else:
        s.save(filename, scalebar=True)
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


def test_export_scalebar_reciprocal(tmp_path):
    pixels = 512
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    for axis in s.axes_manager.signal_axes:
        axis.units = '1/nm'
        axis.scale = 0.1

    filename = tmp_path / 'test_scalebar_export.jpg'
    s.save(filename, scalebar=True, scalebar_kwds={'location':'lower right'})
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


def test_export_scalebar_undefined_units(tmp_path):
    pixels = 512
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))

    filename = tmp_path / 'test_scalebar_export.jpg'
    s.save(filename, scalebar=True, scalebar_kwds={'location':'lower right'})
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


def test_non_uniform(tmp_path):
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].convert_to_non_uniform_axis()

    filename = tmp_path / 'test_export_size.jpg'
    with pytest.raises(TypeError):
        s.save(filename)


@pytest.mark.skipif(not matplotlib_scalebar_installed,
                    reason='matplotlib_scalebar is not installed')
def test_export_scalebar_different_scale_units(tmp_path):
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].scale = 2

    filename = tmp_path / 'test_export_size.jpg'
    with pytest.raises(ValueError):
        s.save(filename, scalebar=True)

    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].units = 'nm'

    filename = tmp_path / 'test_export_size.jpg'
    with pytest.raises(ValueError):
        s.save(filename, scalebar=True)


@pytest.mark.parametrize('output_size', (512, [512, 512]))
def test_export_output_size(output_size, tmp_path):
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))

    fname = tmp_path / 'test_export_size.jpg'
    s.save(fname, scalebar=True, output_size=output_size)
    s_reload = hs.load(fname)
    assert s_reload.data.shape == (512, 512)


@pytest.mark.parametrize('output_size', (512, (512, 512)))
def test_export_output_size_non_square(output_size, tmp_path):
    pixels = (8, 16)
    s = hs.signals.Signal2D(np.arange(np.multiply(*pixels)).reshape(pixels))

    fname = tmp_path / 'test_export_size_non_square.jpg'
    s.save(fname, output_size=output_size)
    s_reload = hs.load(fname)

    if isinstance(output_size, int):
       output_size = (output_size * np.divide(*pixels), output_size)

    assert s_reload.data.shape == output_size


@pytest.mark.parametrize('output_size', (None, 512))
@pytest.mark.parametrize('aspect', (1, 0.5))
def test_export_output_size_aspect(aspect, output_size, tmp_path):
    pixels = (256, 256)
    s = hs.signals.Signal2D(np.arange(np.multiply(*pixels)).reshape(pixels))

    fname = tmp_path / 'test_export_size_non_square_aspect.jpg'
    s.save(fname, scalebar=True, output_size=output_size, imshow_kwds=dict(aspect=aspect))
    s_reload = hs.load(fname)

    if output_size is None:
        output_size = s.data.shape[0]
    assert s_reload.data.shape == (output_size * aspect, output_size)


def test_save_image_navigation(tmp_path):
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))

    fname = tmp_path / 'test_save_image_navigation.jpg'
    s.T.save(fname, scalebar=True)
