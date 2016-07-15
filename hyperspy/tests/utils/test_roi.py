import nose.tools as nt

import numpy as np

from hyperspy.signals import Signal2D, Signal1D
from hyperspy.roi import (Point1DROI, Point2DROI, SpanROI, RectangularROI,
                          Line2DROI, CircleROI)


class TestROIs():

    def setUp(self):
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
        nt.assert_equal(sr.axes_manager.navigation_shape,
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
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, int(35 / scale), ...])

    def test_point2d_image(self):
        s = self.s_i
        r = Point2DROI(35, 40)
        sr = r(s)
        scale = s.axes_manager[0].scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        s.axes_manager.navigation_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[int(40 / scale), int(35 / scale), ...])

    def test_point2d_image_sig(self):
        s = self.s_i
        r = Point2DROI(1, 2)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        nt.assert_equal(sr.axes_manager.signal_shape,
                        s.axes_manager.signal_shape[2:])
        np.testing.assert_equal(
            sr.data, s.data[..., int(2 / scale), int(1 / scale)])

    def test_span_spectrum_nav(self):
        s = self.s_s
        r = SpanROI(15, 30)
        sr = r(s)
        scale = s.axes_manager[0].scale
        n = (30 - 15) / scale
        nt.assert_equal(sr.axes_manager.navigation_shape,
                        (n, ) + s.axes_manager.navigation_shape[1:])
        np.testing.assert_equal(
            sr.data, s.data[:, int(15 / scale):int(30 // scale), ...])

    def test_span_spectrum_sig(self):
        s = self.s_s
        r = SpanROI(1, 3)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        n = (3 - 1) / scale
        nt.assert_equal(sr.axes_manager.signal_shape, (n, ))
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
        nt.assert_equal(sr.axes_manager.navigation_shape,
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
        nt.assert_equal(sr.axes_manager.navigation_shape, (n, n))
        nt.assert_equal(sr_ann.axes_manager.navigation_shape, (n, n))
        # Check that mask is same for all images:
        for i in range(n):
            for j in range(n):
                nt.assert_true(np.all(sr.data.mask[j, i, :] == True) or
                               np.all(sr.data.mask[j, i, :] == False))
                nt.assert_true(np.all(sr_ann.data.mask[j, i, :] == True) or
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
        nt.assert_equal(np.sum(sr.data), (n**2 - 3 * 4) * 4)
        nt.assert_equal(np.sum(sr_ann.data), 4 * 5 * 4)

    def test_2d_line_spec_plot(self):
        r = Line2DROI(10, 10, 150, 50, 5)
        s = self.s_s
        s2 = r(s)
        np.testing.assert_almost_equal(s2.data, np.array(
            [[0.96779467, 0.5468849, 0.27482357, 0.59223042],
             [0.75928245, 0.36454463, 0.50106317, 0.37638916],
             [0.93916091, 0.50631222, 0.99980858, 0.19725947],
             [0.8075638, 0.05100731, 0.62716071, 0.50245307],
             [0.88118825, 0.91641901, 0.2715511, 0.60754536],
             [0.8996517, 0.11646325, 0.16318171, 0.6962192],
             [0.01497626, 0.4573887, 0.64439714, 0.06037948],
             [0.72362309, 0.28890677, 0.97364152, 0.85953663],
             [0.72362309, 0.28890677, 0.97364152, 0.85953663],
             [0.24783865, 0.1928807, 0.21518258, 0.33911841],
             [0.50438632, 0.55765081, 0.31787285, 0.614175],
             [0.04143613, 0.40084015, 0.79034035, 0.64846836],
             [0.08067386, 0.79973013, 0.16217087, 0.19457452],
             [0.54601389, 0.8121854, 0.81069635, 0.71423997],
             [0.83030369, 0.1979076, 0.10475427, 0.05690645],
             [0.30969465, 0.60847027, 0.76278635, 0.54082552],
             [0.61242728, 0.11588176, 0.60404942, 0.83680764],
             [0.57727334, 0.36670814, 0.04252509, 0.43254571],
             [0.89750871, 0.44035511, 0.6426097, 0.38146028],
             [0.32126274, 0.6520685, 0.47213129, 0.05101303],
             [0.30668186, 0.51515877, 0.69551083, 0.28720246],
             [0.95568239, 0.12785748, 0.77676417, 0.31861386],
             [0.80813283, 0.93128198, 0.28999035, 0.39246662],
             [0.80813283, 0.93128198, 0.28999035, 0.39246662],
             [0.3329224, 0.95850972, 0.8414333, 0.28555722],
             [0.84350086, 0.93051062, 0.82974137, 0.52569971],
             [0.97002278, 0.82518232, 0.92972355, 0.52315373],
             [0.23035238, 0.43235565, 0.04142909, 0.04025669],
             [0.54124516, 0.89630549, 0.5923778, 0.25663082],
             [0.60156, 0.51353417, 0.78213733, 0.8870782],
             [0.08783705, 0.59293514, 0.39825928, 0.55032904]]
        ))
        r.linewidth = 50
        s3 = r(s)
        np.testing.assert_almost_equal(s3.data, np.array(
            [[0.44265776, 0.35745439, 0.27446296, 0.41595865],
             [0.26916342, 0.37715306, 0.32971154, 0.33530845],
             [0.35426084, 0.47282874, 0.44090125, 0.35714432],
             [0.4460463, 0.44976548, 0.40195731, 0.53014602],
             [0.5426024, 0.53311424, 0.54084761, 0.3422711],
             [0.45614115, 0.43188745, 0.25545383, 0.44532886],
             [0.46822001, 0.62948798, 0.3153703, 0.43942564],
             [0.47173014, 0.67098345, 0.25334107, 0.37146341],
             [0.51756393, 0.37361861, 0.43797954, 0.4143497],
             [0.42342347, 0.54374975, 0.63824925, 0.46078933],
             [0.59427019, 0.49465564, 0.50841493, 0.64876998],
             [0.55886454, 0.48120671, 0.53824089, 0.59540268],
             [0.57945739, 0.64865535, 0.44827414, 0.50337253],
             [0.53121262, 0.47216267, 0.55630321, 0.4016824],
             [0.43845159, 0.47476225, 0.40063072, 0.35417257],
             [0.45185363, 0.42844148, 0.5125228, 0.47386537],
             [0.45610424, 0.56858719, 0.60593161, 0.47611862],
             [0.47861857, 0.57419121, 0.51392978, 0.59486733],
             [0.47641588, 0.54820991, 0.52233701, 0.57604036],
             [0.34994652, 0.4514264, 0.57328452, 0.57024109],
             [0.34159438, 0.59016962, 0.56304279, 0.4982093],
             [0.55053388, 0.65012605, 0.55998794, 0.47428968],
             [0.51392427, 0.56827423, 0.74809453, 0.60156568],
             [0.64152041, 0.56110823, 0.57792954, 0.49621454],
             [0.53466093, 0.55640288, 0.42431698, 0.50720362],
             [0.59924828, 0.39606699, 0.56290688, 0.59880921],
             [0.43478479, 0.46415729, 0.41994764, 0.5724431],
             [0.26896758, 0.41500659, 0.47386699, 0.56557835],
             [0.62045775, 0.49978237, 0.62594581, 0.48231989],
             [0.3378933, 0.62744914, 0.61643326, 0.57795806],
             [0.4226275, 0.53682311, 0.58369682, 0.43471438]]
        ))

    def test_2d_line_img_plot(self):
        s = self.s_i
        r = Line2DROI(0, 0, 4, 4, 1)
        s2 = r(s)
        nt.assert_true(np.allclose(s2.data, np.array(
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
        )))
        r.linewidth = 10
        s3 = r(s)
        nt.assert_true(np.allclose(s3.data, np.array(
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
        )))


class TestInteractive:

    def setup(self):
        self.s = Signal1D(np.arange(2000).reshape((20, 10, 10)))

    def test_out(self):
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        sr = r(s)
        d = s.data.sum()
        sr.data += 2
        nt.assert_equal(d + sr.data.size * 2, s.data.sum())
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
