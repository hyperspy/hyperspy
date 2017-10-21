

import numpy as np

from hyperspy.signals import Signal2D, Signal1D
from hyperspy.roi import (Point1DROI, Point2DROI, SpanROI, RectangularROI,
                          Line2DROI, CircleROI)


class TestROIs():

    def setup_method(self, method):
        np.random.seed(0)  # Same random every time, Line2DROi test requires it
        self.s_s = Signal1D(np.random.rand(50, 60, 4))
        self.s_s.axes_manager[0].scale = 5
        self.s_s.axes_manager[0].units = 'nm'
        self.s_s.axes_manager[1].scale = 5
        self.s_s.axes_manager[1].units = 'nm'

        # 4D dataset
        self.s_i = Signal2D(np.random.rand(100, 100, 4, 4))

    def test_point1d_spectrum(self):
        s = self.s_s
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert (sr.axes_manager.navigation_shape ==
                s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, int(35 / scale), ...])

    def test_point1d_spectrum_ronded_coord(self):
        s = self.s_s
        r = Point1DROI(37.)
        sr = r(s)
        scale = s.axes_manager[0].scale
        np.testing.assert_equal(
            sr.data, s.data[:, int(round(37 / scale)), ...])
        r = Point1DROI(39.)
        sr = r(s)
        np.testing.assert_equal(
            sr.data, s.data[:, int(round(39 / scale)), ...])

    def test_point1d_image(self):
        s = self.s_i
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert (sr.axes_manager.navigation_shape ==
                s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, int(35 / scale), ...])

    def test_point2d_image(self):
        s = self.s_i
        r = Point2DROI(35, 40)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert (sr.axes_manager.navigation_shape ==
                s.axes_manager.navigation_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[int(40 / scale), int(35 / scale), ...])

    def test_point2d_image_sig(self):
        s = self.s_i
        r = Point2DROI(1, 2)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        assert (sr.axes_manager.signal_shape ==
                s.axes_manager.signal_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[..., int(2 / scale), int(1 / scale)])

    def test_span_spectrum_nav(self):
        s = self.s_s
        r = SpanROI(15, 30)
        sr = r(s)
        scale = s.axes_manager[0].scale
        n = (30 - 15) / scale
        assert (sr.axes_manager.navigation_shape ==
                (n, ) + s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, int(15 / scale):int(30 // scale), ...])

    def test_span_spectrum_sig(self):
        s = self.s_s
        r = SpanROI(1, 3)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        n = (3 - 1) / scale
        assert sr.axes_manager.signal_shape == (n, )
        np.testing.assert_equal(sr.data, s.data[...,
                                                int(1 / scale):int(3 / scale)])

    def test_rect_image(self):
        s = self.s_i
        s.axes_manager[0].scale = 0.2
        s.axes_manager[1].scale = 0.8
        r = RectangularROI(left=2.3, top=5.6, right=3.5, bottom=12.2)
        sr = r(s)
        scale0 = s.axes_manager[0].scale
        scale1 = s.axes_manager[1].scale
        n = ((int(round(2.3 / scale0)), int(round(3.5 / scale0)),),
             (int(round(5.6 / scale1)), int(round(12.2 / scale1)),))
        assert (sr.axes_manager.navigation_shape ==
                (n[0][1] - n[0][0], n[1][1] - n[1][0]))
        np.testing.assert_equal(
            sr.data, s.data[n[1][0]:n[1][1], n[0][0]:n[0][1], ...])

    def test_circle_spec(self):
        s = self.s_s
        s.data = np.ones_like(s.data)
        r = CircleROI(20, 25, 20)
        r_ann = CircleROI(20, 25, 20, 15)
        sr = r(s)
        sr_ann = r_ann(s)
        scale = s.axes_manager[0].scale
        n = int(round(40 / scale))
        assert sr.axes_manager.navigation_shape == (n, n)
        assert sr_ann.axes_manager.navigation_shape == (n, n)
        # Check that mask is same for all images:
        for i in range(n):
            for j in range(n):
                assert (np.all(sr.data.mask[j, i, :] == True) or
                        np.all(sr.data.mask[j, i, :] == False))
                assert (np.all(sr_ann.data.mask[j, i, :] == True) or
                        np.all(sr_ann.data.mask[j, i, :] == False))
        # Check that the correct elements has been masked out:
        mask = sr.data.mask[:, :, 0]
        print(mask)   # To help debugging, this shows the shape of the mask
        np.testing.assert_array_equal(
            np.where(mask.flatten())[0],
            [0, 1, 6, 7, 8, 15, 48, 55, 56, 57, 62, 63])
        mask_ann = sr_ann.data.mask[:, :, 0]
        print(mask_ann)   # To help debugging, this shows the shape of the mask
        np.testing.assert_array_equal(
            np.where(mask_ann.flatten())[0],
            [0, 1, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 25,
             26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46,
             48, 50, 51, 52, 53, 55, 56, 57, 62, 63])
        # Check that mask works for sum
        assert np.sum(sr.data) == (n**2 - 3 * 4) * 4
        assert np.sum(sr_ann.data) == 4 * 5 * 4

    def test_2d_line_spec_plot(self):
        r = Line2DROI(10, 10, 150, 50, 5)
        s = self.s_s
        s2 = r(s)
        np.testing.assert_allclose(s2.data, np.array(
            [[0.96779467, 0.5468849, 0.27482357, 0.59223042],
             [0.89676116, 0.40673335, 0.55207828, 0.27165277],
                [0.27734027, 0.52437981, 0.11738029, 0.15984529],
                [0.04680635, 0.97073144, 0.00386035, 0.17857997],
                [0.61286675, 0.0813696, 0.8818965, 0.71962016],
                [0.96638997, 0.50763555, 0.30040368, 0.54950057],
                [0.22956744, 0.50686296, 0.73685316, 0.09767637],
                [0.5149222, 0.93841202, 0.22864655, 0.67714114],
                [0.5149222, 0.93841202, 0.22864655, 0.67714114],
                [0.59288027, 0.0100637, 0.4758262, 0.70877039],
                [0.80546244, 0.58610794, 0.56928692, 0.51208072],
                [0.97176308, 0.36384478, 0.78791575, 0.55529411],
                [0.39563367, 0.95546593, 0.59831597, 0.11891694],
                [0.4175392, 0.78158173, 0.69374702, 0.91634033],
                [0.44679332, 0.83699037, 0.22182403, 0.49394526],
                [0.92961874, 0.66721471, 0.79807902, 0.55099397],
                [0.98046646, 0.58866215, 0.04551071, 0.1979828],
                [0.70340703, 0.35307496, 0.15442542, 0.31268984],
                [0.88432423, 0.95853234, 0.20751273, 0.78846839],
                [0.27334874, 0.88713154, 0.16554561, 0.66595992],
                [0.08421126, 0.97389332, 0.70063334, 0.84181574],
                [0.15946909, 0.41702974, 0.42681952, 0.26810926],
                [0.13159685, 0.03921054, 0.02523183, 0.27155029],
                [0.13159685, 0.03921054, 0.02523183, 0.27155029],
                [0.46185344, 0.72624328, 0.4748717, 0.90405082],
                [0.52917427, 0.54280647, 0.71405379, 0.51655594],
                [0.13307599, 0.77345467, 0.4062725, 0.96309389],
                [0.28351378, 0.26307878, 0.3335074, 0.57231702],
                [0.89486974, 0.17628164, 0.2796788, 0.58167984],
                [0.64937273, 0.5006921, 0.28355772, 0.2861476],
                [0.31342052, 0.19085, 0.90192363, 0.85839813]]
        ), rtol=0.05)
        r.linewidth = 50
        s3 = r(s)
        np.testing.assert_allclose(s3.data, np.array(
            [[0.40999384, 0.27111487, 0.3345655, 0.47553854],
             [0.44475117, 0.40330205, 0.48113292, 0.26780132],
                [0.57911599, 0.38999298, 0.38509116, 0.37418655],
                [0.29175157, 0.37856367, 0.34420691, 0.48316543],
                [0.55975912, 0.57155145, 0.57640677, 0.39718605],
                [0.41300845, 0.45929259, 0.27489573, 0.40120352],
                [0.46271229, 0.60908378, 0.25796662, 0.46526239],
                [0.37843991, 0.54919334, 0.40469436, 0.48612034],
                [0.44717148, 0.44934708, 0.29064827, 0.51334849],
                [0.3966089, 0.59853786, 0.50392157, 0.39123649],
                [0.50281456, 0.62863149, 0.43051921, 0.32015553],
                [0.40527468, 0.44258442, 0.55694228, 0.41142292],
                [0.47856163, 0.49720026, 0.62012372, 0.47537808],
                [0.46695064, 0.5159018, 0.53532036, 0.4691573],
                [0.44267241, 0.46886762, 0.37363574, 0.54369291],
                [0.76138395, 0.54406653, 0.47305104, 0.45083095],
                [0.74812744, 0.53414434, 0.38487816, 0.44611049],
                [0.59011489, 0.5456799, 0.41782293, 0.5948403],
                [0.47546595, 0.52536805, 0.39267032, 0.58787463],
                [0.39387115, 0.4784124, 0.36765754, 0.46951847],
                [0.54076839, 0.69257203, 0.44540576, 0.39236971],
                [0.41195904, 0.5148879, 0.51199686, 0.63694563],
                [0.44885787, 0.46886977, 0.42150512, 0.52556669],
                [0.60826081, 0.3987657, 0.55875628, 0.5293137],
                [0.44151911, 0.4188617, 0.37734811, 0.51166705],
                [0.52878209, 0.41050467, 0.57149806, 0.52577575],
                [0.50474464, 0.3294767, 0.63519013, 0.56126315],
                [0.37607782, 0.58086952, 0.45089019, 0.62929377],
                [0.59956085, 0.5173887, 0.64790597, 0.49865165],
                [0.57646846, 0.46468029, 0.45267259, 0.44889072],
                [0.4382186, 0.49576157, 0.6192481, 0.45031413]]
        ))

    def test_2d_line_img_plot(self):
        s = self.s_i
        r = Line2DROI(0, 0, 4, 4, 1)
        s2 = r(s)
        assert np.allclose(s2.data, np.array(
            [[[0.5646904, 0.83974605, 0.37688365, 0.499676],
              [0.08130241, 0.3241552, 0.91565131, 0.85345237],
              [0.5941565, 0.90536555, 0.42692772, 0.93761072],
              [0.9458708, 0.56996783, 0.05020319, 0.88466194]],

             [[0.55342858, 0.71776076, 0.9698018, 0.84684608],
              [0.77676046, 0.32998726, 0.49284904, 0.63849364],
              [0.94969472, 0.99393561, 0.79184028, 0.60493951],
              [0.99584095, 0.83632682, 0.51592399, 0.53049253]],

             [[0.55342858, 0.71776076, 0.9698018, 0.84684608],
              [0.77676046, 0.32998726, 0.49284904, 0.63849364],
              [0.94969472, 0.99393561, 0.79184028, 0.60493951],
              [0.99584095, 0.83632682, 0.51592399, 0.53049253]],

             [[0.32270396, 0.28878038, 0.64165074, 0.92820531],
              [0.24836647, 0.37477366, 0.18406007, 0.11019336],
              [0.38678734, 0.9174347, 0.47658793, 0.45095935],
              [0.95232706, 0.96468026, 0.5158903, 0.69112322]],

             [[0.72414297, 0.64417135, 0.17938658, 0.12279276],
              [0.90632348, 0.90345183, 0.21473533, 0.34087282],
              [0.2579504, 0.65663038, 0.27606922, 0.33695786],
              [0.46466925, 0.34991125, 0.73593611, 0.32203574]],

             [[0.72414297, 0.64417135, 0.17938658, 0.12279276],
              [0.90632348, 0.90345183, 0.21473533, 0.34087282],
              [0.2579504, 0.65663038, 0.27606922, 0.33695786],
              [0.46466925, 0.34991125, 0.73593611, 0.32203574]],

             [[0.97259866, 0.13527587, 0.48531393, 0.31607768],
              [0.13656701, 0.40578067, 0.64221493, 0.46036815],
              [0.30466093, 0.88706533, 0.30914269, 0.01833664],
              [0.56143007, 0.09026307, 0.81898535, 0.4518825]]]
        ))
        r.linewidth = 10
        s3 = r(s)
        assert np.allclose(s3.data, np.array(
            [[[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]],

             [[0.12385935, 0.17534623, 0.08266437, 0.08533342],
              [0.06072978, 0.18213069, 0.13162582, 0.14526251],
              [0.11950599, 0.09530544, 0.05814531, 0.10613925],
              [0.13243216, 0.13388253, 0.15641767, 0.07678893]],

             [[0.10387718, 0.18591981, 0.21704829, 0.16594489],
              [0.26554947, 0.27280648, 0.23534874, 0.15751378],
              [0.11329239, 0.16440693, 0.19378236, 0.23418843],
              [0.20414672, 0.24669051, 0.08809065, 0.21252996]],

             [[0.32737802, 0.24354627, 0.25713232, 0.42447693],
              [0.22132115, 0.34440789, 0.1769873, 0.18348862],
              [0.32205928, 0.29038094, 0.22570116, 0.20305065],
              [0.45399669, 0.29687212, 0.313637, 0.27469796]],

             [[0.38104394, 0.2654458, 0.51666151, 0.47973295],
              [0.34333797, 0.36907303, 0.34349318, 0.25681538],
              [0.32849871, 0.27963978, 0.47319042, 0.37358476],
              [0.48767599, 0.23022751, 0.32004745, 0.37714935]],

             [[0.59093609, 0.54976286, 0.54934114, 0.54753303],
              [0.48284716, 0.35797562, 0.49739056, 0.46934957],
              [0.29954848, 0.45448276, 0.50639968, 0.56140708],
              [0.55790493, 0.55105139, 0.40859302, 0.47408336]],

             [[0.63293155, 0.38872956, 0.55044015, 0.37731745],
              [0.49091568, 0.54173188, 0.51292652, 0.53813843],
              [0.56463766, 0.73848284, 0.41183566, 0.37515417],
              [0.48426503, 0.23582684, 0.45947953, 0.49322732]]]
        ))


class TestInteractive:

    def setup_method(self, method):
        self.s = Signal1D(np.arange(2000).reshape((20, 10, 10)))

    def test_out(self):
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        sr = r(s)
        d = s.data.sum()
        sr.data += 2
        assert d + sr.data.size * 2 == s.data.sum()
        r.x += 2
        sr2 = r(s)
        r(s, out=sr)
        np.testing.assert_array_equal(sr2.data, sr.data)

    def test_out_special_case(self):
        s = self.s.inav[0]
        r = CircleROI(3, 5, 2)
        sr = r(s)
        np.testing.assert_array_equal(np.where(sr.data.mask.flatten())[0],
                                      [0, 3, 12, 15])
        r.r_inner = 1
        r.cy = 16
        sr2 = r(s)
        r(s, out=sr)
        np.testing.assert_array_equal(np.where(sr.data.mask.flatten())[0],
                                      [0, 3, 5, 6, 9, 10, 12, 15])
        np.testing.assert_array_equal(sr2.data, sr.data)

    def test_interactive_special_case(self):
        s = self.s.inav[0]
        r = CircleROI(3, 5, 2)
        sr = r.interactive(s, None, color="blue")
        np.testing.assert_array_equal(np.where(sr.data.mask.flatten())[0],
                                      [0, 3, 12, 15])
        r.r_inner = 1
        r.cy = 16
        sr2 = r(s)
        np.testing.assert_array_equal(np.where(sr.data.mask.flatten())[0],
                                      [0, 3, 5, 6, 9, 10, 12, 15])
        np.testing.assert_array_equal(sr2.data, sr.data)

    def test_interactive(self):
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        sr = r.interactive(s, None)
        r.x += 5
        sr2 = r(s)
        np.testing.assert_array_equal(sr.data, sr2.data)
