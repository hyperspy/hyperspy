from matplotlib import artist, path as mpath, transforms
from matplotlib.collections import Collection, _CollectionWithSizes
from matplotlib.textpath import TextPath, TextToPath
from matplotlib.font_manager import FontProperties
import numpy as np
import math


class RegularPolyCollection(_CollectionWithSizes):
    """A collection of n-sided regular polygons."""

    _path_generator = mpath.Path.unit_regular_polygon
    _factor = np.pi ** (-1/2)

    def __init__(self,
                 numsides,
                 rotation=0,
                 sizes=(1,),
                 **kwargs):
        """
        Parameters
        ----------
        numsides : int
            The number of sides of the polygon.
        rotation : float
            The rotation of the polygon in radians.
        sizes : tuple of float
            The area of the circle circumscribing the polygon in points^2.
        **kwargs
            Forwarded to `.Collection`.

        Examples
        --------
        See :doc:`/gallery/event_handling/lasso_demo` for a complete example::

            offsets = np.random.rand(20, 2)
            facecolors = [cm.jet(x) for x in np.random.rand(20)]

            collection = RegularPolyCollection(
                numsides=5, # a pentagon
                rotation=0, sizes=(50,),
                facecolors=facecolors,
                edgecolors=("black",),
                linewidths=(1,),
                offsets=offsets,
                offset_transform=ax.transData,
                )
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self._numsides = numsides
        self._paths = [self._path_generator(numsides)]
        self._rotation = rotation
        self.set_transform(transforms.IdentityTransform())

    def get_numsides(self):
        """Returns the number of sides of the polygon."""
        return self._numsides

    def get_rotation(self):
        """Returns the rotation of the polygon in radians."""
        return self._rotation

    def set_rotation(self, rotation):
        """The rotation of the polygon in radians."""
        self._rotation = rotation

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.figure.dpi)
        self._transforms = [
            transforms.Affine2D(x).rotate(-self._rotation).get_matrix()
            for x in self._transforms
        ]
        # Explicitly not super().draw, because set_sizes must be called before
        # updating self._transforms.
        Collection.draw(self, renderer)


class _CollectionWithWidthHeightAngle(Collection):
    """
    Base class for collections that have an array of widths, heights and angles
    """

    def __init__(self, widths, heights, angles, units='points', **kwargs):
        """
        Parameters
        ----------
        widths : array-like
            The lengths of the first axes (e.g., major axis lengths).
        heights : array-like
            The lengths of second axes.
        angles : array-like
            The angles of the first axes, degrees CCW from the x-axis.
        units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
            The units in which majors and minors are given; 'width' and
            'height' refer to the dimensions of the axes, while 'x' and 'y'
            refer to the *offsets* data units. 'xy' differs from all others in
            that the angle as plotted varies with the aspect ratio, and equals
            the specified angle only when the aspect ratio is unity.  Hence
            it behaves the same as the `~.patches.Ellipse` with
            ``axes.transData`` as its transform.
        **kwargs
            Forwarded to `Collection`.
        """
        super().__init__(**kwargs)
        self._set_widths(widths)
        self._set_heights(heights)
        self._set_angles(angles)
        self._units = units
        self.set_transform(transforms.IdentityTransform())
        self._transforms = np.empty((0, 3, 3))
        self._paths = [self._path_generator()]

    def _set_transforms(self):
        """Calculate transforms immediately before drawing."""

        ax = self.axes
        fig = self.figure

        if self._units == 'xy':
            sc = 1
        elif self._units == 'x':
            sc = ax.bbox.width / ax.viewLim.width
        elif self._units == 'y':
            sc = ax.bbox.height / ax.viewLim.height
        elif self._units == 'inches':
            sc = fig.dpi
        elif self._units == 'points':
            sc = fig.dpi / 72.0
        elif self._units == 'width':
            sc = ax.bbox.width
        elif self._units == 'height':
            sc = ax.bbox.height
        elif self._units == 'dots':
            sc = 1.0
        else:
            raise ValueError(f'Unrecognized units: {self._units!r}')

        self._transforms = np.zeros((len(self._widths), 3, 3))
        widths = self._widths * sc
        heights = self._heights * sc
        sin_angle = np.sin(self._angles)
        cos_angle = np.cos(self._angles)
        self._transforms[:, 0, 0] = widths * cos_angle
        self._transforms[:, 0, 1] = heights * -sin_angle
        self._transforms[:, 1, 0] = widths * sin_angle
        self._transforms[:, 1, 1] = heights * cos_angle
        self._transforms[:, 2, 2] = 1.0

        _affine = transforms.Affine2D
        if self._units == 'xy':
            m = ax.transData.get_affine().get_matrix().copy()
            m[:2, 2:] = 0
            self.set_transform(_affine(m))

    def _set_widths(self, widths):
        self._widths = 0.5 * np.asarray(widths).ravel()

    def _set_heights(self, heights):
        self._heights = 0.5 * np.asarray(heights).ravel()

    def _set_angles(self, angles):
        self._angles = np.deg2rad(angles).ravel()

    def set_widths(self, widths):
        """Set the lengths of the first axes (e.g., major axis lengths)."""
        self._set_widths(widths)
        self.stale = True

    def set_heights(self, heights):
        """Set the lengths of second axes.."""
        self._set_heights(heights)
        self.stale = True

    def set_angles(self, angles):
        """Set the angles of the first axes, degrees CCW from the x-axis."""
        self._set_angles(angles)
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        self._set_transforms()
        super().draw(renderer)


class EllipseCollection(_CollectionWithWidthHeightAngle):
    """
    A collection of ellipses, drawn using splines.

    Parameters
    ----------
    widths : array-like
        The lengths of the first axes (e.g., major axis lengths).
    heights : array-like
        The lengths of second axes.
    angles : array-like
        The angles of the first axes, degrees CCW from the x-axis.
    units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
        The units in which majors and minors are given; 'width' and
        'height' refer to the dimensions of the axes, while 'x' and 'y'
        refer to the *offsets* data units. 'xy' differs from all others in
        that the angle as plotted varies with the aspect ratio, and equals
        the specified angle only when the aspect ratio is unity.  Hence
        it behaves the same as the `~.patches.Ellipse` with
        ``axes.transData`` as its transform.
    **kwargs
        Forwarded to `Collection`.
    """

    _path_generator = mpath.Path.unit_circle


class RectangleCollection(_CollectionWithWidthHeightAngle):
    """
    A collection of rectangles, drawn using splines.

    Parameters
    ----------
    widths : array-like
        The lengths of the first axes (e.g., major axis lengths).
    heights : array-like
        The lengths of second axes.
    angles : array-like
        The angles of the first axes, degrees CCW from the x-axis.
    units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
        The units in which majors and minors are given; 'width' and
        'height' refer to the dimensions of the axes, while 'x' and 'y'
        refer to the *offsets* data units. 'xy' differs from all others in
        that the angle as plotted varies with the aspect ratio, and equals
        the specified angle only when the aspect ratio is unity.  Hence
        it behaves the same as the `~.patches.Ellipse` with
        ``axes.transData`` as its transform.
    **kwargs
        Forwarded to `Collection`.

    """
    _path_generator = mpath.Path.unit_rectangle


class TextCollection(Collection):
    _factor = 1.0
    def __init__(self,
                 texts,
                 sizes=None,
                 rotation=0,
                 horizontalalignment='center',
                 verticalalignment='center',
                 prop=None,
                 usetex=False,
                 **kwargs):
        """
        Parameters
        ----------
        texts : array-like
            The texts of the collection.
        angles : array-like
            The angles of the first axes, degrees CCW from the x-axis.
        **kwargs
            Forwarded to `Collection`.
        """
        super().__init__(**kwargs)
        if sizes is None:
            sizes = [1]
        self._sizes = sizes
        self._horizontalalignment = horizontalalignment
        self._verticalalignment = verticalalignment
        self._transforms = np.empty((0, 3, 3))  # for rotating and shifting the text
        self.rotation = rotation
        self.usetex = usetex
        if prop is None:
            self.prop = FontProperties()
        else:
            self.prop = prop
        self._set_texts(texts)



    def _set_texts(self, texts):
        self._texts = texts
        self._generate_path_from_text()

    def set_texts(self, texts):
        self._set_texts(texts)
        self.stale = True

    def get_texts(self, texts):
        return self._texts

    def set_rotation_center_and_sizes(self, dpi=72.0):
        """
        Calculate transforms immediately before drawing.
        """
        self._transforms = np.zeros((len(self._texts), 3, 3))
        scales = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
        scales = [scales[i % len(self._sizes)] for i in range(len(self._texts))]
        self._transforms[:, 0, 0] = scales  # set the size of the text in x
        self._transforms[:, 1, 1] = scales  # set the size of the text in y
        self._transforms[:, 2, 2] = 1.0

        text_to_path = TextToPath()
        for i, t in enumerate(self._texts):
            width, height, decent = text_to_path.get_text_width_height_descent(t,
                                                                               prop=self.prop,
                                                                               ismath=self.usetex)

            translation_ = [0, 0]
            if self._horizontalalignment == 'center':
                translation_[0] = -width/2 * self._transforms[i][0, 0]
            elif self._horizontalalignment == 'left':
                translation_[0] = 0
            elif self._horizontalalignment == 'right':
                translation_[0] = -width * self._transforms[i][0, 0]
            else:
                raise ValueError(f'Unrecognized horizontalalignment: {self._horizontalalignment!r}')

            if self._verticalalignment == 'center':
                translation_[1] = -height/2 * self._transforms[i][0, 0]
            elif self._verticalalignment == 'bottom':
                translation_[1] = 0
            elif self._verticalalignment == 'top':
                translation_[1] = -height * self._transforms[i][0, 0]
            else:
                raise ValueError(f'Unrecognized verticalalignment: {self._verticalalignment!r}')
            translation = [0, 0]
            translation[1] = math.sin(self.rotation)*translation_[0] + math.cos(self.rotation)*translation_[1]
            translation[0] = math.cos(self.rotation)*translation_[0] - math.sin(self.rotation)*translation_[1]
            self._transforms[i] = translate_matrix(rotate_matrix(self._transforms[i], self.rotation),
                                                   translation[0],
                                                   translation[1])
            self.stale = True

    def _generate_path_from_text(self):
        # For each TextPath, the position is at (0, 0) because the position
        # will be given by the offsets values
        self._paths = [TextPath((0, 0), text, prop=self.prop, usetex=self.usetex) for text in self._texts]

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_rotation_center_and_sizes(self.figure.dpi)
        super().draw(renderer)

def rotate_matrix(mat, theta):
    a = math.cos(theta)
    b = math.sin(theta)
    mtx = mat
    # Operating and assigning one scalar at a time is much faster.
    (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
    # mtx = [[a -b 0], [b a 0], [0 0 1]] * mtx
    mtx[0, 0] = a * xx - b * yx
    mtx[0, 1] = a * xy - b * yy
    mtx[0, 2] = a * x0 - b * y0
    mtx[1, 0] = b * xx + a * yx
    mtx[1, 1] = b * xy + a * yy
    mtx[1, 2] = b * x0 + a * y0
    return mtx

def translate_matrix(mat, tx, ty):
    mat[0, 2] += tx
    mat[1, 2] += ty
    return mat
