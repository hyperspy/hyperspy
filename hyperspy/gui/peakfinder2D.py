import inspect
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from hyperspy._signals.signal2d import find_peaks_max, find_peaks_minmax, \
    find_peaks_zaefferer, find_peaks_stat, find_peaks_dog, find_peaks_log

METHODS = [find_peaks_max, find_peaks_minmax, find_peaks_zaefferer,
           find_peaks_stat, find_peaks_dog, find_peaks_log]


class PeakFinderUIBase:

    def __init__(self):
        self.signal = None
        self.indices = None
        self.methods = METHODS
        self.method_names = [method.__name__ for method in METHODS]
        self.params = {method.__name__: OrderedDict(
            [(p.name, p.default) for p in
             inspect.signature(method).parameters.values() if
             p.default is not inspect._empty]) for method in METHODS}
        self._method = self.method_names[0]

    def interactive(self, signal):
        self.signal = signal
        self.indices = self.signal.axes_manager.indices
        self.init_ui()

    def init_ui(self):
        raise NotImplementedError

    @property
    def current_method(self):
        return dict(zip(self.method_names, self.methods))[self._method]

    def get_data(self):
        return self.signal.inav[self.indices].data

    def get_peaks(self):
        peaks = self.current_method(self.get_data(),
                                    **self.params[self._method])
        return peaks


class PeakFinderUIIPYW(PeakFinderUIBase):
    """
    Find peaks using a Jupyter notebook-based user interface
    """

    def __init__(self):
        super(PeakFinderUIIPYW, self).__init__()
        self.ax = None
        self.image = None
        self.pts = None
        self.param_container = None

    def init_ui(self):
        self.create_choices_widget()
        self.create_navigator()
        self.create_param_widgets()
        self.plot()

    def create_choices_widget(self):
        from ipywidgets import Dropdown
        dropdown = Dropdown(
            options=list(self.method_names),
            value=self._method,
            description="Method",
        )

        def on_method_change(change):
            self._method = dropdown.value
            self.create_param_widgets()
            self.replot_peaks()

        dropdown.observe(on_method_change, names='value')
        display(dropdown)

    def create_navigator(self):
        from ipywidgets import HBox
        container = HBox()
        if self.signal.axes_manager.navigation_dimension == 2:
            container = self.create_navigator_2d()
        elif self.signal.axes_manager.navigation_dimension == 1:
            container = self.create_navigator_1d()
        display(container)

    def create_navigator_1d(self):
        import ipywidgets as ipyw
        x_min, x_max = 0, self.signal.axes_manager.navigation_size - 1
        x_text = ipyw.BoundedIntText(value=self.indices[0],
                                     description="Coordinate", min=x_min,
                                     max=x_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        randomize = ipyw.Button(description="Randomize",
                                layout=ipyw.Layout(flex='0 1 auto',
                                                   width='auto'))
        container = ipyw.HBox((x_text, randomize))

        def on_index_change(change):
            self.indices = (x_text.value,)
            self.replot_image()

        def on_randomize(change):
            from random import randint
            x = randint(x_min, x_max)
            x_text.value = x

        x_text.observe(on_index_change, names='value')
        randomize.on_click(on_randomize)
        return container

    def create_navigator_2d(self):
        import ipywidgets as ipyw
        x_min, y_min = 0, 0
        x_max, y_max = self.signal.axes_manager.navigation_shape
        x_max -= 1
        y_max -= 1
        x_text = ipyw.BoundedIntText(value=self.indices[0], description="x",
                                     min=x_min, max=x_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        y_text = ipyw.BoundedIntText(value=self.indices[1], description="y",
                                     min=y_min, max=y_max,
                                     layout=ipyw.Layout(flex='0 1 auto',
                                                        width='auto'))
        randomize = ipyw.Button(description="Randomize",
                                layout=ipyw.Layout(flex='0 1 auto',
                                                   width='auto'))
        container = ipyw.HBox((x_text, y_text, randomize))

        def on_index_change(change):
            self.indices = (x_text.value, y_text.value)
            self.replot_image()

        def on_randomize(change):
            from random import randint
            x = randint(x_min, x_max)
            y = randint(y_min, y_max)
            x_text.value = x
            y_text.value = y

        x_text.observe(on_index_change, names='value')
        y_text.observe(on_index_change, names='value')
        randomize.on_click(on_randomize)
        return container

    def create_param_widgets(self):
        from ipywidgets import VBox
        containers = []
        if self.param_container:
            self.param_container.close()
        for param, value in self.params[self._method].items():
            container = self.create_param_widget(param, value)
            containers.append(container)
        self.param_container = VBox(containers, description="Parameters")
        display(self.param_container)

    def create_param_widget(self, param, value):
        from ipywidgets import Layout, HBox
        children = (HBox(),)
        if isinstance(value, bool):
            from ipywidgets import Label, ToggleButton
            p = Label(value=param, layout=Layout(width='10%'))
            t = ToggleButton(description=str(value), value=value)

            def on_bool_change(change):
                t.description = str(change['new'])
                self.params[self._method][param] = change['new']
                self.replot_peaks()

            t.observe(on_bool_change, names='value')

            children = (p, t)

        elif isinstance(value, float):
            from ipywidgets import FloatSlider, FloatText, BoundedFloatText, \
                Label
            from traitlets import link
            p = Label(value=param, layout=Layout(flex='0 1 auto', width='10%'))
            b = BoundedFloatText(value=0, min=1e-10,
                                 layout=Layout(flex='0 1 auto', width='10%'),
                                 font_weight='bold')
            a = FloatText(value=2 * value,
                          layout=Layout(flex='0 1 auto', width='10%'))
            f = FloatSlider(value=value, min=b.value, max=a.value,
                            step=np.abs(a.value - b.value) * 0.01,
                            layout=Layout(flex='1 1 auto', width='60%'))
            l = FloatText(value=f.value,
                          layout=Layout(flex='0 1 auto', width='10%'),
                          disabled=True)
            link((f, 'value'), (l, 'value'))

            def on_min_change(change):
                if f.max > change['new']:
                    f.min = change['new']
                    f.step = np.abs(f.max - f.min) * 0.01

            def on_max_change(change):
                if f.min < change['new']:
                    f.max = change['new']
                    f.step = np.abs(f.max - f.min) * 0.01

            def on_param_change(change):
                self.params[self._method][param] = change['new']
                self.replot_peaks()

            b.observe(on_min_change, names='value')
            f.observe(on_param_change, names='value')
            a.observe(on_max_change, names='value')
            children = (p, l, b, f, a)

        elif isinstance(value, int):
            from ipywidgets import IntSlider, IntText, BoundedIntText, \
                Label
            from traitlets import link
            p = Label(value=param, layout=Layout(flex='0 1 auto', width='10%'))
            b = BoundedIntText(value=0, min=1e-10,
                               layout=Layout(flex='0 1 auto', width='10%'),
                               font_weight='bold')
            a = IntText(value=2 * value,
                        layout=Layout(flex='0 1 auto', width='10%'))
            f = IntSlider(value=value, min=b.value, max=a.value,
                          step=np.abs(a.value - b.value) * 0.01,
                          layout=Layout(flex='1 1 auto', width='60%'))
            l = IntText(value=f.value,
                        layout=Layout(flex='0 1 auto', width='10%'),
                        disabled=True)
            link((f, 'value'), (l, 'value'))

            def on_min_change(change):
                if f.max > change['new']:
                    f.min = change['new']
                    f.step = np.abs(f.max - f.min) * 0.01

            def on_max_change(change):
                if f.min < change['new']:
                    f.max = change['new']
                    f.step = np.abs(f.max - f.min) * 0.01

            def on_param_change(change):
                self.params[self._method][param] = change['new']
                self.replot_peaks()

            b.observe(on_min_change, names='value')
            f.observe(on_param_change, names='value')
            a.observe(on_max_change, names='value')
            children = (p, l, b, f, a)
        container = HBox(children)
        return container

    def plot(self):
        self.ax = None
        self.plot_image()
        self.plot_peaks()

    def plot_image(self):
        if self.ax is None:
            self.ax = plt.figure().add_subplot(111)
        z = self.get_data()
        self.image = self.ax.imshow(np.rot90(np.fliplr(z)))
        self.ax.set_xlim(0, z.shape[0])
        self.ax.set_ylim(0, z.shape[1])
        plt.show()

    def replot_image(self):
        if not plt.get_fignums():
            self.plot()
        z = self.get_data()
        self.image.set_data(np.rot90(np.fliplr(z)))
        self.replot_peaks()
        plt.draw()

    def plot_peaks(self):
        peaks = self.get_peaks()
        self.pts, = self.ax.plot(peaks[:, 0], peaks[:, 1], 'o')
        plt.show()

    def replot_peaks(self):
        if not plt.get_fignums():
            self.plot()
        peaks = self.get_peaks()
        self.pts.set_xdata(peaks[:, 0])
        self.pts.set_ydata(peaks[:, 1])
        plt.draw()
