# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy.signal import Signal


class Image(Signal):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Image, self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0 + 3j)

    def tomographic_reconstruction(self,
                                   algorithm='FBP',
                                   tilt_stages='auto',
                                   iteration=1,
                                   parallel=None,
                                   **kwargs):
        """
        Reconstruct a 3D tomogram from a sinogram

        Parameters
        ----------
        algorithm: {'FBP','SART'}
            FBP, filtered back projection
            SART, Simultaneous Algebraic Reconstruction Technique
        tilt_stages: list or 'auto'
            the angles of the sinogram. If 'auto', takes axes_manager
        iteration: int
            The numebr of iteration used for SART
        parallel : {None, int}
            If None or 1, does not parallelise multifit. If >1, will look for
            ipython clusters. If no ipython clusters are running, it will
            create multiprocessing cluster.

        Return
        ------
        The reconstruction as a 3D image

        Examples
        --------
        >>> adf_tilt = database.image3D('tilt_TEM')
        >>> adf_tilt.change_dtype('float')
        >>> rec = adf_tilt.tomographic_reconstruction()
        """
        from hyperspy._signals.spectrum import Spectrum
        # import time
        if parallel is None:
            sinogram = self.to_spectrum().data
        if tilt_stages == 'auto':
            tilt_stages = self.axes_manager[0].axis
        # a = time.time()
        if algorithm == 'FBP':
            # from skimage.transform import iradon
            from skimage.transform import iradon
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon(sinogram[i], theta=tilt_stages,
                                output_size=sinogram.shape[1], **kwargs)
        elif algorithm == 'SART' and parallel is None:
            from skimage.transform import iradon_sart
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                     **kwargs)
                for j in range(iteration - 1):
                    rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                         image=rec[i], **kwargs)
        elif algorithm == 'SART':
            from hyperspy.misc import multiprocessing
            pool, pool_type = multiprocessing.pool(parallel)
            sino = multiprocessing.split(self.to_spectrum(), parallel, axis=1)
            kwargs.update({'theta': tilt_stages})
            data = [[si.data, iteration, kwargs] for si in sino]
            res = pool.map_sync(multiprocessing.isart, data)
            if pool_type == 'mp':
                pool.close()
                pool.join()
            # res = res.get()
            rec = res[0]
            for i in range(len(res)-1):
                rec = np.append(rec, res[i+1], axis=0)

        # print time.time() - a

        rec = Spectrum(rec).as_image([2, 1])
        rec.axes_manager = self.axes_manager.deepcopy()
        rec.axes_manager[0].scale = rec.axes_manager[1].scale
        rec.axes_manager[0].offset = rec.axes_manager[1].offset
        rec.axes_manager[0].units = rec.axes_manager[1].units
        rec.axes_manager[0].name = 'z'
        rec.get_dimensions_from_data()
        return rec
