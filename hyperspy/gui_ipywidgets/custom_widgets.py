import ipywidgets
import traitlets


class OddIntSlider(ipywidgets.IntSlider):

    @traitlets.validate('value')
    def _validate_value(self, proposal):
        value = proposal['value']
        if not self.value % 2:
            value += 1
        if self.min > value or self.max < value:
            value = min(max(value, self.min), self.max)
        return value
